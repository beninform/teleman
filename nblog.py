import logging
from datetime import datetime


class NBLog():
    def __init__(self):
        self.ts = datetime.now()
        # construct new run string
        self.newrun = f"{'-'*25} NEW RUN {'-'*25}"
        

# create logger
logger = logging.getLogger(__name__)
# set log level for all handlers to debug
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# create file handler and set level to debug
fileHandler = logging.FileHandler('run.log')
fileHandler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to handlers
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

# set log level for development
logger.setLevel(logging.DEBUG)

