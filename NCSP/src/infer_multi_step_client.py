


import argparse
from multiprocessing import Manager
import multiprocessing
import os

import concurrent.futures
import traceback

from src.infer_one_step_client import InferOneStepClient
from src.utils import get_function_from_file, is_process_running, validate_function_signature, load_data, save_data, get_hash, list_files, check_and_create_dir, CommonClient, get_un_generated_data
from src.logging_config import logger_manager
logger = logger_manager(0)


def run_each_step(args, _step2pid, _last_step_datas, _generated_datas):
    step_client = InferOneStepClient(args)
    try:
        step_client(_step2pid, _last_step_datas, _generated_datas)
    except KeyboardInterrupt as e:
        raise Exception(">>> My KeyboardInterrupt Exception")
    except Exception as e:
        logger.error(f"step {args.step_idx} error: {e}")
        _step2pid[args.step_idx] = -1
        traceback.print_exc()
        raise e

class InferMultiStepClient:
    def __init__(self, args_list) -> None:
        self.args_list = args_list
        self.Manager = Manager()

 
        self._generated_datas = []
        for idx, args in enumerate(self.args_list):
            self._generated_datas.append(multiprocessing.Manager().list())
        
        self._last_step_datas_of_step0 = multiprocessing.Manager().list()
        
        self._step2pid = self.Manager.dict()


    def run(self):
        processes = []  

        for idx, args in enumerate(self.args_list):

            _last_step_datas = self._generated_datas[idx - 1] if idx > 0 else self._last_step_datas_of_step0
            process = multiprocessing.Process(target=run_each_step,
                                              args=(args, self._step2pid, _last_step_datas, self._generated_datas[idx]))
            processes.append(process)
            process.start()  


        for process in processes:
            process.join()  


            if process.exitcode != 0:
                logger.error(f"one process closed with exit code {process.exitcode}")
            else:
                logger.info(f"process {process.name} completed successfully.")
            
            del process  



