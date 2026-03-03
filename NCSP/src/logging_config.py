import logging
import os
import shutil
import time


class Logger_Manager:
    def __init__(self):
        self.logger_dict = {}
        self.cur_time = os.getenv('TIMESTAMP', time.strftime('%Y%m%d%H%M%S', time.localtime()))
        self.cur_day = self.cur_time[:8]
        self.cur_second = self.cur_time[8:]
        self.log_dir = f'./log/{self.cur_day}'

    def get_logger(self, idx):
        # 
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 
        if idx in self.logger_dict:
            return self.logger_dict[idx]
        else:
            logger = logging.getLogger(f'S{idx}')
            
            logger.setLevel(logging.DEBUG)

            # 
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            #
            fh = logging.FileHandler(os.path.join(self.log_dir, f'{self.cur_second}_step{idx}.log'))
            fh.setLevel(logging.DEBUG)

            # 
            ch_formatter = logging.Formatter('\r%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(ch_formatter)
            fh.setFormatter(fh_formatter)

            # 
            logger.addHandler(ch)
            logger.addHandler(fh)

            self.logger_dict[idx] = logger
            return logger
    def __call__(self, idx):
        return self.get_logger(idx)


logger_manager = Logger_Manager()
