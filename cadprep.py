import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from datetime import datetime

# note time of cadprep run
cadprepts = datetime.now()

class CleanCAD():
    def __init__(self, df, dq=False):
        self.dq  = dq
        self.incd_cat_type = None
        self.MPDS_cat_type = None
        self.cdf = self.process(df)
        self.ts = datetime.now()
    
    def start_cleaning(self, dfx):
        dfy = dfx.copy()
        return dfy

    def clean_values(self, dfy):
        dfy.columns = [c.replace(" ", "") for c in dfy]  # strip spaces out of column headers
        return dfy

    def check_cat_data(self, dfy):
        """
        Defines categorical data types and checks incoming data for conformity to the listed categories.
        """
        # define category types for sortability
        incd_cats = ['RED', 'AMBER', 'GREEN', 'Routine', 'Cat A', 'Cat C']
        self.incd_cat_type = CategoricalDtype(categories=incd_cats, ordered=True)
        
        MPDS_cats = ['RED', 'AMBER1', 'AMBER2', 'GREEN2', 'GREEN3', 'AS3 - Not Applicable', 'RED1', 'RED2', 'GREEN1']
        self.MPDS_cat_type = CategoricalDtype(categories=MPDS_cats, ordered=True)
        
        # check for alignment with category value expectations (ie, new category levels we didn't know about)
        # check alignment for IncidentCategory
        new_cat1 = np.unique(dfy[~dfy['IncidentCategory'].isin(incd_cats)]['IncidentCategory']).tolist()
        new_cat_count = len(new_cat1)
        if len(new_cat1) > 0:
            print('WARNING ! ! ! ! - You have at least one category that is not catered for in the field "IncidentCategory"')
            print('Total new categories:', new_cat_count)
            print(new_cat1)

        # correct the known instance of changed orthography
        dfy['MPDSPriorityType'] = dfy['MPDSPriorityType'].str.replace('Red2','RED2')
        
        # check alignment for MPDSPriorityType
        new_cat2 = np.unique(dfy[~dfy['MPDSPriorityType'].isin(MPDS_cats)]['MPDSPriorityType']).tolist()
        new_cat_count = len(new_cat2)
        if len(new_cat2) > 0:
            print('WARNING ! ! ! ! - You have at least one category that is not catered for in the field "MPDSPriorityType"')
            print('Total new categories:', new_cat_count)
            print(new_cat2)

        return dfy
        

    def change_data_types(self, dfy):
        dfy['IncdDate'] = pd.to_datetime(dfy.IncidentDate, format='%d/%m/%Y')  # proper datetime
        dfy['IncdTime'] = pd.to_timedelta(dfy.IncidentTime)  # proper time
        dfy['VMobBPDt'] = pd.to_datetime(dfy.VehicleMobileButtonPressDateTime, format='%d/%m/%Y %H:%M')  # proper datetime
        dfy['VArrASDt'] = pd.to_datetime(dfy.VehicleArrivalAtSceneDateTime, format='%d/%m/%Y %H:%M')  # proper datetime
        dfy['VAllocDt'] = pd.to_datetime(dfy.VehicleAllocatedDateTime, format='%d/%m/%Y %H:%M')  # proper datetime
        return dfy

    def impute_values(self, dfy):
        dfy.DispatchCode = dfy.DispatchCode.fillna('00O00')
        return dfy

    def derive_useful_col_values(self, dfy):
        
        dfy['MPDSlen'] = dfy.DispatchCode.str.len()
        dfy['prot_cde'] = dfy.DispatchCode.str[:2]
        dfy['detA_cde'] = dfy.DispatchCode.str[2:]
        dfy['detnt'] = dfy.detA_cde.str[:1]
        dfy['subdt'] = dfy.detA_cde.str[1:]  # will capture suffix letters, if they appear (ie, all chars after 3rd char)
        dfy['prot_cat'] = pd.Categorical(dfy.prot_cde)
        
        dfy['iYear'] = dfy.IncdDate.dt.year
        dfy['iMnth'] = dfy.IncdDate.dt.month
        dfy['iDyoM'] = dfy.IncdDate.dt.day
        dfy['iDyoW'] = dfy.IncdDate.dt.day_name().str[:3]        
        dfy['iDnoW'] = dfy.IncdDate.dt.dayofweek        
        
        dfy['MnthNo'] = ('0'+dfy.monthorder.astype('str')).str.strip().str[-2:]
        dfy['YrMnth'] = dfy.Year.map(str)+dfy.MnthNo
        
        dfy['NoIraw'] = dfy.NatureOfIncident.astype('str')
        
        NoIlist = [('0'+str(n))[-2:] for n in range(40)]  # make a list of number strings '00' to '39'
        dfy['NoINo'] = np.where(dfy['NatureOfIncident'].isin(NoIlist), dfy.NatureOfIncident, '00').astype('str')  # a field for the number strings
        dfy['NoICd'] = np.where(dfy['NatureOfIncident'].isin(NoIlist), 'XXXXX', dfy.NatureOfIncident).astype('str')  # a field for the char strings
        
        dfy['PtAgeBin'] = pd.cut(dfy[['AgeOfPatient']].values.flatten(), [10*i for i in range(13)])  # upper limit is 130 years
        
        dfy['pc_outer'] = dfy['postcodemapping']
        dfy['pc_inner'] = dfy['PostCode-Copy.2']
        dfy['pc_inn1'] = dfy['pc_inner'].str[:1]

        dfy['VAL'] = pd.notnull(dfy.VAllocDt)
        dfy['MBP'] = pd.notnull(dfy.VMobBPDt)
        dfy['VAS'] = pd.notnull(dfy.VArrASDt)        
        dfy['NDP'] = dfy.VAL & ~dfy.MBP
        dfy['ISD'] = dfy.MBP & ~dfy.VAS
        dfy['PrgLbl'] = 'XXX'
        # dfy['DelRsn'] = 0    
        

        # need cat type in order to be sortable
        dfy['incd_cat'] = dfy.IncidentCategory.astype(self.incd_cat_type)

        # need cat type in order to be sortable
        dfy['MPDSPrt'] = dfy.MPDSPriorityType.astype(self.MPDS_cat_type)
        
        # create new col values for each incident (row count and min dt)
        incidentsdf = (dfy.groupby('IncidentID')
                       .agg(
                           rpinid=('DispatchCode', np.size), 
                           ivallocdt=('VAllocDt', np.nanmin),
                           ivmobbpdt=('VMobBPDt', np.nanmin),
                           ivarrasdt=('VArrASDt', np.nanmin)
                       ))
        # join the count of rows per incident and min datetime back to dfy
        dfyy = dfy.set_index('IncidentID').join(incidentsdf)
        
        # Create 'dispatchesdf1' to add row count for each dispatch (incident-vehicle)
        dispatchesdf1 = (dfy.groupby(['IncidentID', 'VehicleID'])
                         .agg(
                             rpivid=('DispatchCode', np.size),
                             dvallocdt=('VAllocDt', np.nanmin),
                             dvmobbpdt=('VMobBPDt', np.nanmin),
                             dvarrasdt=('VArrASDt', np.nanmin)
                         ))
        
        
        # join the incident-vehicle data back to the CAD data
        dfyz = dfyy.set_index(['VehicleID'], append=True).join(dispatchesdf1)  #.join(dispatchesdf2)        
        
        # copy data to proceed
        dfzz = dfyz.copy()
        
        # populate ProgLbl (Progress Labels)
        dfzz.loc[ dfzz.VAL & ~dfzz.MBP & ~dfzz.VAS, 'PrgLbl'] = 'Al____'  # TFF
        dfzz.loc[ dfzz.VAL &  dfzz.MBP & ~dfzz.VAS, 'PrgLbl'] = 'AlMb__'  # TTF
        dfzz.loc[ dfzz.VAL &  dfzz.MBP &  dfzz.VAS, 'PrgLbl'] = 'AlMbAt'  # TTT

        dfzz.loc[ dfzz.VAL & ~dfzz.MBP &  dfzz.VAS, 'PrgLbl'] = 'TFT'  # ever seen? # TFT
        dfzz.loc[~dfzz.VAL &  dfzz.MBP &  dfzz.VAS, 'PrgLbl'] = 'FTT'  # ever seen? # FTT
        dfzz.loc[~dfzz.VAL &  dfzz.MBP & ~dfzz.VAS, 'PrgLbl'] = 'FTF'  # ever seen? # FTF
        dfzz.loc[~dfzz.VAL & ~dfzz.MBP &  dfzz.VAS, 'PrgLbl'] = 'FFT'  # ever seen? # FFT
        dfzz.loc[~dfzz.VAL & ~dfzz.MBP & ~dfzz.VAS, 'PrgLbl'] = 'FFF'  # ever seen? # FFF        

        Errdf = dfzz[~dfzz.PrgLbl.isin(['Al____', 'AlMb__', 'AlMbAt'])]
        Errcnt = len(Errdf)
        if Errcnt > 0:
            print(Errcnt, 'CAD rows with illogical timestamps')
            
        dfzz['PrgRnk'] = dfzz.groupby(level=[0,1])['PrgLbl'].rank(method='first').astype('int64')
        
        dfzz.loc[dfzz.PrgRnk==1,'DQ'] = 1
        dfzz.loc[dfzz.PrgRnk!=1,'DQ'] = 0
        dfzz.DQ = dfzz.DQ.astype('int64')
                
        return dfzz
    
    def remove_bad_rows(self, dfy):
        if self.dq:
            dfy1 = dfy.loc[dfy.DQ==1,:]
            return dfy1
        else:
            return dfy
        
    def count_incident_outcomes(self, dfy):
        
        dfy1 = dfy.loc[dfy.DQ==1, :]
        
        dfAND = dfy1[dfy1.PrgLbl=='Al____'].groupby(['IncidentID', 'PrgLbl']).size().to_frame(name='cntAlND').droplevel(1)

        dfSTD = dfy1[dfy1.PrgLbl=='AlMb__'].groupby(['IncidentID', 'PrgLbl']).size().to_frame(name='cntStDn').droplevel(1)

        dfATT = dfy1[dfy1.PrgLbl=='AlMbAt'].groupby(['IncidentID', 'PrgLbl']).size().to_frame(name='cntAttd').droplevel(1)
        
        dfaa = dfy.join(dfAND).join(dfSTD).join(dfATT)
        
        dfaa[['cntAttd', 'cntStDn', 'cntAlND']] = dfaa[['cntAttd', 'cntStDn', 'cntAlND']].fillna(value=0).astype('int64')
        
        return dfaa
        
        

    def process(self, dfin):
        return (dfin.pipe(self.start_cleaning)
                .pipe(self.clean_values)
                .pipe(self.check_cat_data)
                .pipe(self.change_data_types)
                .pipe(self.impute_values)
                .pipe(self.derive_useful_col_values)
                .pipe(self.remove_bad_rows)
                .pipe(self.count_incident_outcomes)
                )
    

class SubSetCAD():
    
    def __init__(self, df, exclyr=None, inclyr=None, collist=None, supcollist=None, exclh=None):
        self.indf = df
        self.exclyr = exclyr
        self.inclyr = inclyr
        self.collist = collist
        self.supcollist = supcollist
        self.exclh = exclh
        self.year_list_raw = np.unique(df.Year).tolist()
        self.year_list_use = self.year_list_raw  # initialise with the raw list
        self.sdf = self.process(df)
        self.ts = datetime.now()
        
    def start_subsetting(self, dfx):
        dfy = dfx.copy()
        return dfy

    def filter_to_recent(self, dfy):
        # Oct-2015 and onwards avoids old codes 'RED1' and 'RED2'
        
        if self.inclyr:
            if not isinstance(self.inclyr, list):
                self.inclyr = [self.inclyr]
            self.year_list_use = self.inclyr
            dfy = dfy[dfy.Year.isin(self.inclyr)]
        elif self.exclyr:
            if self.exclyr in self.year_list_raw:
                self.year_list_use.remove(self.exclyr)
                dfy = dfy[dfy.Year.isin(self.year_list_use)]
            else:
                raise Exception("Sorry, your exclusion year is not in the dataset")
        return dfy

    def trim_to_default_cols(self, data):
        keep = [
            # 'IncidentID', 
            # 'IncidentDate',
            'IncdDate',
            'Year',
            # 'MnthNo',
            # 'YrMnth',
            # 'IncidentCategory', 
            # 'incd_cat',
            # 'MPDSPriorityType', 
            # 'MPDSPrt',
            # 'DispatchCode',
            # 'MPDSlen',
            # 'prot_cde',
            # 'prot_cat',
            # 'detnt',
            # 'subdt',
            # 'MonthShort',
            # 'VehicleID', 
            # 'VehicleType',
            'VAllocDt', 
            'VMobBPDt',
            'VArrASDt',
            'IsStoodDown',
            'rpinid',
            'rpivid',
            'ivallocdt',
            'ivmobbpdt',
            'ivarrasdt',
            'dvallocdt',
            'dvmobbpdt',
            'dvarrasdt',
            'VAL',
            'MBP',
            'VAS',
            'NDP',
            'ISD',
            # 'cntStDnY',
            # 'cntDsStDn',
            # 'cntAtt',
            'PrgLbl',
            'PrgRnk',
            'cntAlND', 
            'cntStDn', 
            'cntAttd', 
            'DQ'
            # 'DelRsn'
            # 'monthorder', 
            # 'WAAcount',
            # 'WAABASE'
        ]
        if self.collist:
            if self.collist=='FULL':
                return data
            else:
                keep = self.collist
        if self.supcollist:
            keep = keep + self.supcollist
        data = data[keep]
        return data
    
    def drop_hosp_transfers(self, df1):
        if self.exclh:
            df2 = df1[df1.IncidentLocationTypeCode!='H'].copy()
            return df2
        else:
            return df1
    
    def process(self, dfin):
        return (dfin.pipe(self.start_subsetting)
                .pipe(self.filter_to_recent)
                .pipe(self.trim_to_default_cols)
                .pipe(self.drop_hosp_transfers)
                )
    
    
class CADheatmap():

    def __init__(self, df, topvar, sidvar, title=None, cmap=None, pltwd=None, pltht=None):
        self.df = df
        self.topvar = topvar
        self.sidvar = sidvar
        
        if cmap:
            self.cmap = cmap
        else:
            self.cmap = 'Blues'
            
        if pltwd:
            self.pltwd=pltwd 
        else:
            self.pltwd=30

        if pltht:
            self.pltht=pltht 
        else:
            self.pltht=10

        if self.topvar and self.sidvar:
            dfA = df.groupby(
                [
                    topvar,
                    sidvar
                ]
                , sort=True).size().reset_index(name='')
            dfA = dfA.set_index([
                topvar, 
                sidvar
            ])
            dfAT = dfA.unstack(level=0)
        elif self.sidvar:
            dfA = df.groupby(sidvar, sort=True).size().to_frame(name='')
            dfAT = dfA
        elif self.topvar:
            dfA = df.groupby(topvar, sort=True).size().to_frame(name='')
            dfAT = dfA.T
        
        fig, ax = plt.subplots(figsize=(self.pltwd, self.pltht))
        sns.heatmap(dfAT, cmap=self.cmap, 
                    # vmin=0, vmax=2000, 
                    annot=True, fmt='4g',
                    square=True,
                    linewidth=0.3,
                    cbar=False
                    # ,cbar_kws={"shrink": 0.8}
                   )
        # xticks
        ax.xaxis.tick_top()

        # yticks
        ax.yaxis.tick_left()

        # axis labels
        plt.xlabel('')
        plt.ylabel('')

        # title
        if not title:
            title = 'Heatmap\n'.upper()
        title = title+'\n'
        plt.title(title, loc='left')
        plt.show()
