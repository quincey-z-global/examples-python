import os
import logging
from logging.handlers import TimedRotatingFileHandler


PATH = os.path.dirname(__file__)


class Logger(object):
    '''
    log writer
    '''

    def __init__(self, log_name: str):
        '''
        log writer, initialise

        @param log_name:  the file name of log
        '''

        self.log_name = log_name
        self.log = logging.getLogger(name=self.log_name)
        self.log.setLevel(level=logging.INFO)

        # initialise logging
        logging.basicConfig()

        # output format
        fmt_st = '%(asctime)s[%(levelname)s][%(processName)s][%(threadName)s]:%(message)s'
        formatter = logging.Formatter(fmt=fmt_st)

        # log file
        path_root = os.path.join(PATH, 'log')  # root directory of log
        if not os.path.exists(path=path_root):
            os.makedirs(name=path_root)
        path = os.path.join(path_root, self.log_name)

        # level and format
        file_handler = TimedRotatingFileHandler(filename=path)
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(fmt=formatter)
        self.log.addHandler(hdlr=file_handler)

    def info(self, msg: str):
        '''
        add normal info

        @param msg: info message
        '''

        self.log.info(msg=' ' + msg)

    def error(self, msg: str):
        '''
        add error info

        @param msg: error message
        '''

        self.log.error(msg=' ' + msg)

        raise Exception(msg)


logger = Logger(log_name='example.log')
