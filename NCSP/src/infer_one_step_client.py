import json
import sys
import time

import aiohttp
import httpx
from openai import OpenAI, AsyncOpenAI
# from critic import async_critic

from collections import defaultdict
import os
from copy import deepcopy
from tqdm import tqdm
import argparse
import concurrent.futures
# from multiprocessing import Lock, Manager
import multiprocessing
import asyncio
from aiohttp import request
# import aiomultiprocess
from tqdm.asyncio import tqdm_asyncio
import random
# import jsonlines
import traceback
from src.mcts import  MCTS
from src.utils import get_function_from_file, is_process_running, multi_process_load_data, preprocess_attach_id_data, split_data, validate_function_signature, load_data, save_data, get_hash, list_files, check_and_create_dir, CommonClient, get_un_generated_data, weighted_random_choice
from src.logging_config import logger_manager


class InferOneStepClient:
    def __init__(self, args):
        self.args = args
        self.logger = logger_manager(args.step_idx)
        # self.ungen_datas = get_un_generated_data(args)

    # 1. get response
    async def api_request(self, args, asyclients, data):
        generate_config = data['generate_config'] if 'generate_config' in data else args.generate_config
        common_config = {}
        extra_body = {}
        for prop in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'best_of', 'n', 'stop', 'stream']:
            if prop in generate_config:
                common_config[prop] = generate_config[prop]
        if 'stream' not in common_config:
            common_config['stream'] = False
        for k,v in generate_config.items():
            if k not in common_config:
                extra_body[k] = v

        if hasattr(args, 'resp_api_weights'):
            random_idx = weighted_random_choice(list(range(len(asyclients))), args.resp_api_weights)
        else:
            random_idx = random.randint(0, len(asyclients)-1)
        if args.mode == 'wait':
            await asyncio.sleep(5)
            resps = ['wait']
            finish_reasons = ['wait']
            stop_reasons = [None]
            if random.random() < 0.01:
                raise Exception('no connection available now')
        elif args.mode == 'chat' or args.mode == 'reasoning_chat':
            resp = await asyclients[random_idx].chat.completions.create(
                messages= data['messages'],
                model=args.resp_server_names[random_idx],
                # stream=stream,
                **common_config,
                extra_body = extra_body
            )
            if common_config['stream'] == False:
                resps = [c.message.content for c in resp.choices]
                if args.mode == 'reasoning_chat' and hasattr(resp.choices[0].message, 'reasoning_content'):
                    reasoning_content = [c.message.reasoning_content for c in resp.choices]
                    resps = {'resp': resps, 'reasoning_content': reasoning_content}
                finish_reasons = [c.finish_reason for c in resp.choices]
                if 'stop_reason' in resp.choices[0]:
                    stop_reasons = [c.stop_reason for c in resp.choices]
                elif 'stop_reason' in resp.choices[0].model_extra:
                    stop_reasons = [c.model_extra['stop_reason'] for c in resp.choices]
                else:
                    stop_reasons = [None] * len(resps)
            else:
                full_content = ""
                async for chunk in resp:
                    if chunk.choices:
                        full_content += chunk.choices[0].delta.content
                resps = [full_content]
                finish_reasons = chunk.choices[0].finish_reason
                if 'stop_reason' in chunk.choices[0]:
                    stop_reasons = [chunk.choices[0].stop_reason]
                elif 'stop_reason' in chunk.choices[0].model_extra:
                    stop_reasons = [chunk.choices[0].model_extra['stop_reason']]
                else:
                    stop_reasons = [None]
            if hasattr(resp, 'model') and hasattr(resp, 'usage') and hasattr(resp.usage, 'prompt_tokens') and hasattr(resp.usage, 'completion_tokens') and hasattr(resp.usage, 'total_tokens'):
                self.logger.info(f"model:{resp.model}, p_token:{resp.usage.prompt_tokens}, r_token:{resp.usage.completion_tokens}, total_token:{resp.usage.total_tokens}")
        elif args.mode == 'completion':
            resp = await asyclients[random_idx].completions.create(
                prompt=data['prompt'],
                model=args.resp_server_names[random_idx],
                stream=False,
                extra_body = generate_config
            )
            resps = [c.text for c in resp.choices]
            finish_reasons = [c.finish_reason for c in resp.choices]
            if 'stop_reason' in resp.choices[0]:
                stop_reasons = [c.stop_reason for c in resp.choices]
            elif 'stop_reason' in resp.choices[0].model_extra:
                stop_reasons = [c.model_extra['stop_reason'] for c in resp.choices]
            else:
                stop_reasons = [None] * len(resps)
        elif args.mode == 'tgi':
            # print(data['prompt'])
            headers = {"Content-Type": "application/json",}
            post_request ={
                "stream": False,
                "inputs": data['prompt'],
                "parameters": generate_config,
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(-1)) as session:
                url = args.resp_urls[random_idx]
                async with session.post(url, headers=headers, json=post_request, timeout=args.api_timeout) as response:
                    if response.headers.get('Content-Type') == 'application/json':
                        response_json = await response.json()
                        details = response_json.get("details", {})
                        if 'error' in response_json:
                            raise Exception(response_json)
                    else:
                        raise Exception(response.text())
                    resps = [response_json["generated_text"]]
                    finish_reasons = [details.get("finish_reason", None)]
                    stop_reasons = [details.get("tokens", [None])[-1]]
        elif args.mode == 'mcts':
            mcts = MCTS(args.mcts_config, asyclients)
            resps, finish_reasons = await mcts.built_tree(data) # return a tree
            stop_reasons = [None]
        else:
            raise Exception(f"mode {args.mode} is not supported, choose from ['chat', 'reasoning_chat', 'completion', 'wait', 'mcts', 'tgi']")
        return resps, finish_reasons, stop_reasons
    
    # 2. filter response
    def resp_filter(self, data, resps, finish_reasons, stop_reasons, resp_filter_func, error_dict):
        resps = resp_filter_func(data, resps, finish_reasons, stop_reasons)
        if resps:
            if self.args.mode in ['chat', 'reasoning_chat', 'completion', 'wait', 'mcts', 'tgi']:
                data['responses'].extend(resps)
        else:
            for resp in resps:
                if resp: resp = resp.replace('\n', '\\n')
                self.logger.error(f"resp verification failed, resp: {resp}")
                error_dict['verify failed'] += 1
                # _last_step_datas.append(data)
        return resps
    
    # 3. save data if enough
    def save_to_file_and_memory(self, data, writer, _generated_datas, has_next_step, tot_request_n, tot_task_n, tot_finish_n, is_save_to_memory=True):
        writer.write(json.dumps(data, ensure_ascii=False)+'\n')
        if has_next_step and is_save_to_memory:
            # _generated_datas.put(data)
            _generated_datas.append(data)
        if tot_finish_n.value % self.args.coroutine_per_process == 0:
            writer.flush()
        if tot_finish_n.value % 100 == 0:
            self.logger.info(f"finish/task = {tot_finish_n.value}/{tot_task_n.value}, request = {tot_request_n.value}")

    async def async_response(self, args, datas, pidx, _step2pid, _last_step_datas, _generated_datas, tot_finish_request_n, tot_request_n, tot_task_n, tot_finish_n, process_status,  last_tot_finish_request_n, last_time):
        asyclients = [AsyncOpenAI(base_url=url, timeout= httpx.Timeout(args.api_timeout, read=args.api_timeout, write=args.api_timeout, connect=args.api_timeout), api_key=api_key) for url, api_key in zip(args.resp_urls, args.resp_api_keys)]
        error_dict = defaultdict(int)
        sleep_time = 1
        has_next_step = args.step_idx != len(_step2pid)-1
        preprocess_func = get_function_from_file(args.preprocess_func, 'main')

        async def coroutine_level_executor(didx, data, writer, semaphore, resp_filter_func):

            nonlocal sleep_time
            async with semaphore:
                if 'responses' not in data:
                    data['responses'] = []
                # if len(data['responses']) >= args.duplicate_num:
                #     return
                try_time = args.max_try_per_request * args.duplicate_num
                while True:
                    # 3. save data if enough
                    try:
                        if len(data['responses']) >= args.duplicate_num:
                            tot_finish_n.value += 1
                            self.save_to_file_and_memory(data, writer, _generated_datas, has_next_step, tot_request_n, tot_task_n, tot_finish_n)
                            return
                    except Exception as e:
                        self.logger.error(f"P{pidx}: save data error: {e}")
                        self.logger.error(traceback.format_exc())
                        return
                    
                    try_time -= 1
                    # 0. error finish condition
                    if try_time < 0: 
                        if len(data['responses']) > 0: 
                            self.save_to_file_and_memory(data, writer, _generated_datas, has_next_step, tot_request_n, tot_task_n, tot_finish_n, is_save_to_memory=False)
                        self.logger.error(f"try_time < 0, exit")
                        return
                    
                    # 1. get response
                    try:
                        if args.debug: self.logger.debug(f"P{pidx}D{didx}: start request api")
                        try:
                            tot_request_n.value += 1
                            resps, finish_reasons, stop_reasons = await self.api_request(args, asyclients, data)
                            tot_finish_request_n.value += 1
                        except Exception as e:
                            tot_finish_request_n.value += 1
                            raise e
                        if args.debug: self.logger.debug(f"P{pidx}D{didx}: request success, responses: {resps}, finish_reasons: {finish_reasons}, stop_reasons: {stop_reasons}, prompt={data['prompt'] if args.mode in ['completion', 'tgi', 'mcts'] else data['messages']}")
                        sleep_time = 1  
                        if tot_request_n.value % 1000 == 0:
                            self.logger.info(f"running/tot request: {tot_request_n.value - tot_finish_request_n.value}/{tot_request_n.value}, QPS: {(tot_finish_request_n.value - last_tot_finish_request_n.value)/(time.time()-last_time.value):.1f}")
                            last_tot_finish_request_n.value = tot_finish_request_n.value
                            last_time.value = time.time()
                            
                    except Exception as e:
                        # a. 
                        str_e = str(e).lower() + '\n' + traceback.format_exc()
                        if 'no connection available now' in str_e:
                            error_dict['no connection available'] += 1
                            # _last_step_datas.append(data)
                        elif 'timed out' in str_e or 'timeout' in str_e:
                            sleep_time -= 1
                            error_dict['timeout'] += 1
                            str_e = "\n".join(str_e.split('\n')[-2:])
                            # _last_step_datas.append(data)
                        elif '上游负载已饱和' in str_e:
                            error_dict['上游负载已饱和'] += 1

                        # b. 
                        elif 'please reduce the length' in str_e: 
                            error_dict['length'] += 1
                            self.logger.error(f"{str_e}")
                            return 
                        elif 'Input validation error' in str_e: 
                            error_dict['Input validation error'] += 1
                            self.logger.error(f"{str_e}")
                            return
                        
                        # c. 
                        else:
                            self.logger.error(f"{str_e}")
                            error_dict['other error'] += 1

                        sleep_time += 1
                        sleep_time = min(sleep_time, 10)
                        self.logger.error(f'request error: {str_e} \n P{pidx}: sleep {sleep_time}s, error: {dict(error_dict)}')
                        await asyncio.sleep(sleep_time)
                        continue
                        
                    # 2. filter response
                    try:
                        resps_copy = deepcopy(resps)
                        resps = self.resp_filter(data, resps, finish_reasons, stop_reasons, resp_filter_func, error_dict)
                        if not resps:
                            error_dict['All has been filted'] += 1
                        if args.debug: 
                            prompt = data['prompt'] if args.mode in ['completion', 'tgi', 'mcts'] else data['messages'][0]['content']
                            self.logger.debug(f"P{pidx}: resps: {resps_copy}, verify result: {resps}")
                            # 
                            if resps and args.mode != 'wait': 
                                with open(os.path.join(args.output_folder, f"md_{args.step_idx}.md"), 'a') as f:
                                    prompt = prompt.replace('|', '\|').replace('>','\>').replace('<','\<').replace('\n', '<br>')
                                    resps_ = '<br>'.join(resps).replace('|', '\|').replace('>','\>').replace('<','\<').replace('\n', '<br>')
                                    tables = f"| ### key | ### val |\n| ---- | ---- |\n| id | {data[args.uniq_key]} |\n| ---- | ---- |\n| prompt | {prompt} |\n| responses | {resps_} |\n| finish_reasons | {str(finish_reasons)} |\n| stop_reasons | {str(stop_reasons)} |\n\n"
                                    f.write(tables)
                                
                    except Exception as e:
                        self.logger.error(f"P{pidx}: verify error: {e}")
                        self.logger.error(traceback.format_exc())
                        continue
      
                
        def get_min_block_data(datas):
            block_datas = []
            # block_size = max(args.coroutine_per_process, datas.qsize()//args.process_num)
            block_size = max(args.coroutine_per_process, len(datas)//args.process_num)
            # block_size = args.coroutine_per_process
            for _ in range(block_size):
                # if datas.qsize() == 0:
                if len(datas) == 0:
                    break
                try:
                    # block_datas.append(datas.get(block=False, timeout=1))
                    block_datas.append(datas.pop())
                except Exception as e:
                    err_msg = f"{e}\n{traceback.format_exc()}"
                    if 'empty' not in err_msg.lower():
                        self.logger.error(f"get_min_block_data error: {err_msg}")
                    break
            if len(block_datas) != 0:
                block_datas = preprocess_func(block_datas)
                tot_task_n.value += len(block_datas)
                self.logger.info(f"finish/task = {tot_finish_n.value}/{tot_task_n.value}, running/tot request: {tot_request_n.value - tot_finish_request_n.value}/{tot_request_n.value}, QPS: {(tot_finish_request_n.value - last_tot_finish_request_n.value)/(time.time()-last_time.value):.1f}")
            return block_datas

        semaphore = asyncio.Semaphore(args.coroutine_per_process)
        resp_filter_func = get_function_from_file(args.resp_filter_func, 'main')

        # 
        with open(os.path.join(args.temp_output_folder, f"{pidx}.jsonl"), 'a') as writer:
            try:
                while True:
                    # step0 
                    if args.step_idx != 0 and len(datas) == 0:
                        datas = get_min_block_data(_last_step_datas)
                    if len(datas) != 0:
                        process_status[pidx] = 1
                        tasks = [coroutine_level_executor(didx, data, writer, semaphore, resp_filter_func) for didx, data in enumerate(datas)]
                        await tqdm_asyncio.gather(*tasks, desc=f"S{args.step_idx}P{pidx}", ncols=60, position=args.step_idx, leave=False)
                        datas = []
                    elif args.step_idx == 0 or not is_process_running(_step2pid[args.step_idx-1]):
                        self.logger.info(f"pre step {args.step_idx-1} has been finished, exit")
                        break
                    else:
                        process_status[pidx] = 2 # 2 means sleeping
                        # 
                        sleep_pidx = [str(pidx_) for pidx_, ps in enumerate(process_status) if ps == 2]
                        if str(pidx) == min(sleep_pidx):
                            self.logger.info(f"P{'P'.join(sleep_pidx)} wait for data; finish/task = {tot_finish_n.value}/{tot_task_n.value}, running/tot request: {tot_request_n.value - tot_finish_request_n.value}/{tot_request_n.value}, QPS: {(tot_finish_request_n.value - last_tot_finish_request_n.value)/(time.time()-last_time.value):.1f}")
                        await asyncio.sleep(10)
            except KeyboardInterrupt:
                self.logger.error(f"Process {pidx} KeyboardInterrupt, writing data to file now !!!\n Do not interrupt the process, wait for it to finish")
                writer.flush()
                raise
            
        self.logger.info(f"P{pidx} done! error_dict: {dict(error_dict)}")
        self.logger.info(f"P{pidx} done! finish/task = {tot_finish_n.value}/{tot_task_n.value}, running/tot request: {tot_request_n.value - tot_finish_request_n.value}/{tot_request_n.value}, QPS: {(tot_finish_request_n.value - last_tot_finish_request_n.value)/(time.time()-last_time.value):.1f}")
        return

    def process_level_executor(self, args, datas, pidx, _step2pid, _last_step_datas, _generated_datas, process_status, tot_finish_request_n,tot_request_n, tot_task_n, tot_finish_n, last_tot_finish_request_n, last_time):
        

        self.logger.info(f"Process {pidx} start")
        try:
            asyncio.run(self.async_response(args, datas, pidx, _step2pid, _last_step_datas, _generated_datas, tot_finish_request_n, tot_request_n, tot_task_n, tot_finish_n, process_status,  last_tot_finish_request_n, last_time))
            process_status[pidx] = 0 # 0 means finished，1 means running，-1 means error
            self.logger.info(f"Process {pidx} done\nrunning process: {sum([ps == 1 for ps in process_status])}/{len(process_status)}\n")
        except KeyboardInterrupt:
            raise Exception(f">>> My KeyboardInterrupt Exception")
        except Exception as e:
            process_status[pidx] = -1
            self.logger.error(f"Process {pidx} error exit: {e} \nrunning process: {sum([ps == 1 for ps in process_status])}/{len(process_status)}\n")
            self.logger.error(traceback.format_exc())
            raise e

    
    def update_data(self, args, _last_step_datas, _generated_datas, tot_task_n, tot_finish_n):
        cur_time = time.time()
        # 1. load from file
        # preprocess -> attach id ->  get ungenerated data from file
        un_generated_datas, tot_finish_n.value = get_un_generated_data(args)

        # 2. load data from shared variable
        data_from_memory = []
        # new_data_num = _last_step_datas.qsize()
        new_data_num = len(_last_step_datas)
        for i in range(new_data_num):
            # data_from_memory.append(_last_step_datas.get(timeout=5))
            data_from_memory.append(_last_step_datas.pop())
        # preprocess -> attach id ->  get ungenerated data from memory
        preprocess_func = get_function_from_file(args.preprocess_func, 'main')
        data_from_memory = preprocess_attach_id_data(args, preprocess_func, data_from_memory)
        un_generated_datas.extend(data_from_memory)

        # self.logger.info(f"Datas generated in memory, last step = {new_data_num}, cur step = {_generated_datas.qsize()}")
        self.logger.info(f"Datas generated in memory, last step = {new_data_num}, cur step = {len(_generated_datas)}")
        # deduplicate
        un_generated_datas = list({data[args.uniq_key]: data for data in un_generated_datas}.values())
        tot_un_generated_n = len(un_generated_datas)
        tot_task_n.value = len(un_generated_datas) + tot_finish_n.value
        # split into blocks
        un_generated_datas = split_data(args, un_generated_datas)
        self.logger.info(f"S{args.step_idx} cost {(time.time()-cur_time)/60:.2f}min load data, get total ungen data: {tot_un_generated_n}")
        
        return un_generated_datas



    def run(self, _step2pid, _last_step_datas, _generated_datas):
        _step2pid[self.args.step_idx] = os.getpid()  

        args = self.args
        start = time.time()
        try_time = 1
        tot_request_n = multiprocessing.Value('i', 0)
        tot_finish_request_n = multiprocessing.Value('i', 0)
        tot_task_n = multiprocessing.Value('i', 0)
        tot_finish_n = multiprocessing.Value('i', 0)
        # 
        last_time = multiprocessing.Value('d', time.time())
        last_tot_finish_request_n = multiprocessing.Value('i', 0)

        processes = []  
        while True:
                self.logger.info(f"\n\n>>>>>>>>>>>>  try_time: {try_time}  <<<<<<<<<<<<<")
                # 1. 
                try:
                    un_generated_datas = self.update_data(args, _last_step_datas, _generated_datas, tot_task_n, tot_finish_n)
                except Exception as e:
                    self.logger.error(f"update_data error: {e}")
                    self.logger.error(traceback.format_exc())
                    raise e
                # 2. 
                if len(un_generated_datas) == 0:
                    if args.step_idx == 0:
                        self.logger.info(f"Step0 all data has been generated")
                        break
                    else:
                        # 
                        if is_process_running(_step2pid[args.step_idx-1]):
                            wait_time = 10
                            self.logger.info(f"pre step {args.step_idx-1} is running, but there is no new data, sleep {wait_time}s")
                            time.sleep(wait_time)
                            try_time += 1
                            continue
                        else:
                            self.logger.info(f"pre step {args.step_idx-1} has been finished, and all data has been generated")
                            break
                # 3.
                else:
                    self.logger.info(f"total un-generated data: {sum([len(datas) for datas in un_generated_datas])}")
                    self.logger.info(f"total concurrent {args.process_num} * {args.coroutine_per_process} = {args.process_num * args.coroutine_per_process}")
                    process_status = multiprocessing.Manager().list()
                    for i in range(args.process_num):
                        process_status.append(1)

                    # 
                    for pidx, datas in enumerate(un_generated_datas):
                        process = multiprocessing.Process(target=self.process_level_executor,
                                                        args=(args, datas, pidx, _step2pid, _last_step_datas, _generated_datas,
                                                                process_status, tot_finish_request_n, tot_request_n,
                                                                tot_task_n, tot_finish_n, last_tot_finish_request_n,
                                                                last_time))
                        processes.append(process)
                        process.start()  # 

                    # 
                    cnt = {}
                    for process in processes:
                        process.join()  # 

                      
                        if process.exitcode != 0:
                            self.logger.error(f"one process closed with exit code {process.exitcode}")
                            cnt['error'] = cnt.get('error', 0) + 1
                        else:
                            cnt['success'] = cnt.get('success', 0) + 1

                    self.logger.info(f"total process: {len(processes)}, {cnt}")
                
                # 4. 
                try_time += 1
                if try_time > args.max_try_per_dataset:
                    self.logger.error(f"try_time > max_try_per_dataset {args.max_try_per_dataset}, exit")
                    break

        
        self.logger.info(f">>> all done {time.time()-start:.2f}s")

        # 
        output_files = list_files(args.temp_output_folder, ext='.jsonl')
        total_datas = multi_process_load_data(output_files)
        
        # 
        id2data = {}
        for data in total_datas:
            if data[args.uniq_key] not in id2data:
                id2data[data[args.uniq_key]] = data
            else:
                if len(data['responses']) > len(id2data[data[args.uniq_key]]['responses']):
                    id2data[data[args.uniq_key]] = data
        total_datas = list(id2data.values())
        
        save_data(total_datas, args.output_path, mode='w')
        self.logger.info(f">>> save {len(total_datas)} data to {args.output_path}")

        # 
        self.logger.info(f"start sleep, waiting for data to be consumed or next step to be finished")
        while True:
            if max(_step2pid.keys()) == args.step_idx:
                self.logger.info(f"all step has finished, exit")
                break
            elif args.step_idx+1 in _step2pid and not is_process_running(_step2pid[args.step_idx+1]):
                self.logger.info(f"next step {args.step_idx+1} has been finished, exit")
                break
            # elif _generated_datas.qsize() == 0:
            elif len(_generated_datas) == 0:
                self.logger.info(f"all data has been consumed, exit")
                break
            time.sleep(10)
        _step2pid[args.step_idx] = -1  # 


    def __call__(self, _step2pid, _last_step_datas, _generated_datas):
        self.run(_step2pid, _last_step_datas, _generated_datas)
            

