import argparse
import asyncio
import concurrent
import hashlib
import json
import mmap
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from multiprocessing import Pool, Process, Queue, Value, cpu_count
from pathlib import Path
from queue import Empty

import pandas as pd
import ujson
from openai import OpenAI
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import importlib.util
import inspect
import psutil
from fractions import Fraction
from typing import get_type_hints

import yaml
from sympy import sympify

from src.logging_config import logger_manager

logger = logger_manager(0)


def run_python(code: str, timeout: int = 20, max_retries: int = 1) -> tuple[Optional[str], Optional[str]]:
    std_err = None
    for _ in range(max_retries):
        try:
            result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=timeout)
            stdout = result.stdout
            stderr = result.stderr
            return stdout, stderr
        except subprocess.TimeoutExpired:
            std_err = "TimeoutExpired"
            continue
        except AttributeError:
            std_err = "AttributeError"
            continue
        except Exception as e:
            std_err = e
            continue
    return None, std_err

def is_code_safe(code):
    for code_line in code.split('\n'):
        code_line = code_line.strip().split(' ')
        if 'import' == code_line[0] or 'from' == code_line[0]:
            code_line = [c.strip() for code in code_line for c in code.split('.')]
            code_line = [c.strip() for code in code_line for c in code.split(',')]
            # print(code_line)
            if any([pkg in code_line for pkg in ['os', 'sys', 'subprocess', 'shutil', 'javax.persistence', 'nacl', 'multiprocessing', 'pyodbc']]):
                logger.warning(f"dangerous code: {code_line}")
                return False
    return True

def weighted_random_choice(items, weights):
    # Calculate the cumulative sum of weights
    cumulative_weights = []
    cumulative_sum = 0
    for weight in weights:
        cumulative_sum += weight
        cumulative_weights.append(cumulative_sum)
    
    # Generate a random number between 0 (inclusive) and cumulative_sum (exclusive)
    rand_num = random.random() * cumulative_sum
    
    # Find the item corresponding to the random number
    for i, cumulative_weight in enumerate(cumulative_weights):
        if rand_num < cumulative_weight:
            return items[i]


def is_process_running(pid):
    """
    检查指定 PID 的进程是否存在。

    :param pid: 进程 ID
    :return: 如果进程存在，返回 True，否则返回 False
    """
    for proc in psutil.process_iter(['pid']):
        if proc.info['pid'] == pid:
            return True
    return False


def is_number_and_is_equal(str1, str2):
    try:
        return True, float(str1) == float(str2)
    except:
        return False, False

def get_function_from_file(file_path, function_name='main'):

    try:
        filename = os.path.basename(file_path)
        module_name, _ = os.path.splitext(filename)


        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        main_function = getattr(module, function_name, None)
        if main_function is None:
            raise ValueError(f"No '{function_name}' function found in {file_path}")
    except Exception as e:
        logger.error(f"\n\nThere was an error loading the function from {file_path}: {e}\n\n")
        raise e

    return main_function


def construct_md_table(path=None, key_list=None, datas=None, save_path=None, length=10):
    table = ''
    if path:
        if os.path.isdir(path):
            paths = list_files(path, ext='.jsonl')
            datas = multi_process_load_data(paths)[:length]
        else:
            datas = load_data(path)[:length]
    else:
        datas = datas[:length]
    if not key_list:
        key_list = list(datas[0].keys())
    for idx, data in enumerate(datas):
        table+=(f'# {idx}\n')
        table+=(f'| key | value |\n')
        table+=(f'| ---- | ---- |\n')
        for key in key_list:
            value = data.get(key, '')
            if key == 'messages': value = "\n\n".join([f"{m['role']}: {m['content']}" for m in value])
            elif type(value) == list: value = '\n'.join(value)
            value = value.replace('|','\|').replace('>','\>').replace('<','\<').replace('\n','<br>')
            table += f'| {key} |  {value} |\n'

        table+=('\n')
    
    if not save_path:
        save_path = path + '.md'
    with open(save_path, 'w') as f:
        f.write(table)
    return table

def validate_function_signature(func, expected_params_example, expected_return_type):


    hints = get_type_hints(func)
    logger.debug(f"函数 {func.__name__} 期望的参数类型: {expected_params_example}")
    logger.debug(f"函数 {func.__name__} 实际参数类型提示: {hints}")

    params = list(hints.values())
    for exp_p, p in zip(expected_params_example, params[:-1]):
        if type(exp_p) != p:
            logger.error(f"函数 {func.__name__} 的参数类型不正确, 期望: {type(exp_p)}, 实际: {p}")
            return False
    
    return True

class CommonClient():
    """
    Sample from TGI's completion API
    """

    def __init__(
        self,
        url: str = "https://127.0.0.1:8080/v1",
        model: str = "glm-4-public",
        api_key: str = 'None',
        system_message: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 1,
        top_k: int = -1,
    ):
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

        self.url = url
        self.model = model

        os.environ["OPENAI_API_KEY"] = api_key if api_key else "test"
        
        self.client = OpenAI(base_url=self.url, timeout=360)

    async def __call__(self, message_list, config: dict = {}, max_try_time = 1) -> str:
        if self.system_message:
            message_list = [{"role": "system", "content": self.system_message}] + message_list

        for i in range(max_try_time):
            try:
                stream = await self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model,
                    stream=False,
                    **config
                )
                output = ''
                # for part in stream:
                #     output += part.choices[0].delta.content
                output = stream.choices[0].message.content
                return output
            except Exception as e:
                logger.error(f'call try_time={i}, request error: {e}')
                time.sleep(1)
                continue
        return ''


def get_hash(obj):
    if isinstance(obj, dict):
        obj_str = json.dumps(obj, sort_keys=True)
        return hashlib.sha256(obj_str.encode('utf-8')).hexdigest()
    elif isinstance(obj, list):
        return hashlib.sha256(json.dumps(obj).encode('utf-8')).hexdigest()
    else:
        return hashlib.sha256(str(obj).encode('utf-8')).hexdigest()

def list_files(directory, ext='.jsonl', recursive=False):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
        if not recursive:
            break
    return files

def check_and_create_dir(path, is_file=True):
    if is_file:
        directory = os.path.dirname(path)
    else:
        directory = path

    if not os.path.exists(directory):
        logger.warning(f"{directory} does not exist, creating...")
        os.makedirs(directory)



class AdvancedJsonlReader:

    
    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or 8
        self.line_count = 0
        
    def _find_next_newline(self, mm: mmap.mmap, start: int) -> int:

        pos = mm.find(b'\n', start)
        return pos + 1 if pos != -1 else mm.size()
    
    def _worker_process(self, start_pos: int, end_pos: int, 
                       filepath: str, queue: Queue, error_count: Value) -> None:

        results = []
        try:
            with open(filepath, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0) as mm:
                    if start_pos > 0:
                        start_pos = self._find_next_newline(mm, start_pos)
                    
                    if end_pos < mm.size():
                        end_pos = self._find_next_newline(mm, end_pos)
                    
                    mm.seek(start_pos)
                    
                    chunk_size = end_pos - start_pos
                    data = mm.read(chunk_size).decode('utf-8')
                    lines = data.split('\n')
                    
                    for line in tqdm(lines, ncols=100, 
                                   desc=f"Processing {Path(filepath).name}", 
                                   position=0, leave=False):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            results.append(ujson.loads(line))
                        except Exception as e:
                            with error_count.get_lock():
                                error_count.value += 1
                            logger.warning(f": {str(e)}")
                            
        except Exception as e:
            logger.error(f": {str(e)}")
        finally:
            queue.put(results)
    
    def read_single_file(self, filepath: str) -> tuple:
        """"""
        file_size = Path(filepath).stat().st_size
        chunk_size = file_size // self.num_processes
        
        queue = Queue()
        processes = []
        error_count = Value('i', 0)
        
        start_time = time.time()
        for i in range(self.num_processes):
            start_pos = i * chunk_size
            end_pos = file_size if i == self.num_processes - 1 else (start_pos + chunk_size)
            
            p = Process(target=self._worker_process,
                       args=(start_pos, end_pos, filepath, queue, error_count))
            processes.append(p)
            p.start()
        
        results = []
        for _ in range(self.num_processes):
            try:
                chunk_results = queue.get(timeout=60)
                results.extend(chunk_results)
            except Empty:
                logger.error(f" {filepath} ")
        
        for p in processes:
            p.join()
            
        file_line_count = len(results) + error_count.value
        logger.info(f"File: {Path(filepath).name}, Time: {(time.time()-start_time):.2f}s, "
                   f"Total lines: {file_line_count}, Errors: {error_count.value}, "
                   f"Results: {len(results)}")

        return results, error_count.value

    def _process_files_parallel(self, filepaths: List[str]) -> List[Dict[Any, Any]]:
        """"""
        queue = Queue()
        processes = []
        
        # 启动处理每个文件的进程
        for filepath in filepaths:
            if not Path(filepath).exists():
                logger.error(f": {filepath}")
                continue
                
            p = Process(target=lambda q, f: q.put(self.read_single_file(f)), 
                       args=(queue, filepath))
            processes.append(p)
            p.start()
        
        # 收集结果
        all_results = []
        total_errors = 0
        for _ in range(len(processes)):
            try:
                results, errors = queue.get(timeout=3600)  # 设置较长的超时时间
                all_results.extend(results)
                total_errors += errors
            except Empty:
                logger.error("")
        
        # 等待所有进程结束
        for p in processes:
            p.join()
            
        return all_results, total_errors
    
    def __call__(self, filepaths: Union[str, List[str]]) -> List[Dict[Any, Any]]:
        """"""
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        
        all_results, total_errors = self._process_files_parallel(filepaths)
        self.line_count = len(all_results) + total_errors
        return all_results


def load_data(dataset_path, input_type=None, as_type='json'):
    if input_type is None:
        input_type = dataset_path.split('.')[-1]
    data = []
    # start_time = time.time()
    err_count = 0
    if input_type == 'jsonl':
        with open(dataset_path, 'r') as f:
            for idx, line in tqdm(enumerate(f), ncols=100, desc=f"DataLoading ", position=0, leave=False):
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    err_count += 1
                    logger.error(f"json line-{idx} in {dataset_path} is broken")
                    continue
    elif input_type == 'json':
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    elif input_type == 'zip-jsonl':
        import zipfile
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            # 
            first_file = zip_ref.namelist()[0]
            with zip_ref.open(first_file) as f:
                # 
                import io
                text_io = io.TextIOWrapper(f, encoding='utf-8')
                for idx, line in tqdm(enumerate(text_io), ncols=100, desc="DataLoading ", position=0, leave=False):
                    try:
                        data.append(json.loads(line))
                    except Exception as e:
                        err_count += 1
                        logger.error(f"json line-{idx} in {dataset_path} is broken")
                        continue
    elif input_type == 'parquet':
        data = pd.read_parquet(dataset_path)
        if as_type == 'json':
            data = data.to_dict(orient='records')
    elif input_type == 'txt':
        with open(dataset_path, 'r') as f:
            data = f.readlines()
            data = [line for line in data]
    elif input_type == 'csv':
        data = pd.read_csv(dataset_path)
        if as_type == 'json':
            data = data.to_dict(orient='records')
    elif input_type == 'yaml':
        with open(dataset_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            
    else:
        raise ValueError(f'input_type {input_type} is not supported')
    # logger.info(f"time: {(time.time()-start_time):.2f} s, err: {err_count}, result: {len(data)}")
    return data

def multi_process_load_data(paths, input_type='jsonl', as_type='json'):
    data = []
    # 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(load_data, paths))
        for result in results:
            data.extend(result)
    return data

def save_data(data, output_path, output_type='jsonl', mode='w'):
    assert type(data) in [list, dict], 'data should be list or dict'
    if mode == 'w':
        if output_type == 'jsonl':
            with open(output_path, 'w') as f:
                for line in data:
                    f.write(json.dumps(line)+'\n')
        elif output_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f)
        elif output_type == 'xlsx':
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
        elif output_type == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(data, f)
        elif output_type == 'txt':
            with open(output_path, 'w') as f:
                for line in data:
                    f.write(line+'\n')
        elif output_type == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f'output_type {output_type} is not supported')
    elif mode == 'a':
        if output_type == 'jsonl':
            with open(output_path, 'a') as f:
                for line in data:
                    f.write(json.dumps(line)+'\n')
        else:
            raise ValueError(f'output_type {output_type} is not supported')
    else:
        raise ValueError(f'mode {mode} is not supported')


def duplicate_and_attach_id(datas, uniq_key, duplicate_num):
    new_datas = []
    for data in datas:
        for i in range(duplicate_num):
            new_data = deepcopy(data)
            new_data[uniq_key] = get_hash(new_data) + f'_{i:05d}'
            new_datas.append(new_data)
    return new_datas

def preprocess_attach_id_data(args, preprocess_func, datas):
    if datas:
        datas = preprocess_func(datas)
        if args.step_idx == 0:
            for data in datas:
                if args.uniq_key not in data:
                    data[args.uniq_key] = get_hash(data)
        else:
            # 
            uniq_datas = {}
            for data in datas:
                if data[args.uniq_key] not in uniq_datas:
                    uniq_datas[data[args.uniq_key]] = data
                else:
                    if len(uniq_datas[data[args.uniq_key]]['responses']) < len(data['responses']):
                        uniq_datas[data[args.uniq_key]] = data
            datas = list(uniq_datas.values())
    return datas

def split_data(args, un_generated_datas):
    if len(un_generated_datas) == 0:
        return []
    data_split = []
    block_num = args.process_num
    if args.process_num > len(un_generated_datas):
        block_num = len(un_generated_datas)
    size_per_thread = (len(un_generated_datas) + block_num - 1) // block_num
    for i in range(0, len(un_generated_datas), size_per_thread):
        data_split.append(un_generated_datas[i:i+size_per_thread])
    #
    for i in range(len(data_split), args.process_num):
        data_split.append([])
    return data_split

def get_un_generated_data(args):
    start_time = time.time()

    # 1. load
    if os.path.isdir(args.dataset_path):
        input_paths = list_files(args.dataset_path, ext='.jsonl')
    else:
        input_paths = [args.dataset_path]
    datas = multi_process_load_data(input_paths, input_type='jsonl')
                
    if args.debug: datas = datas[:args.debug]

    # 2. preprocess data, duplicate and attach id
    preprocess_func = get_function_from_file(args.preprocess_func, 'main')
    datas = preprocess_attach_id_data(args, preprocess_func, datas)

    # 3. find out ungenerated data
    generate_ids2data = {}
    output_paths = list_files(args.temp_output_folder, ext='.jsonl')
    output_datas = multi_process_load_data(output_paths, input_type='jsonl')
    for data in output_datas:
            if data[args.uniq_key] not in generate_ids2data:
                generate_ids2data[data[args.uniq_key]] = data
            else:
                if len(generate_ids2data[data[args.uniq_key]]['responses']) < len(data['responses']):
                    generate_ids2data[data[args.uniq_key]] = data

    un_generated_datas = []
    for i, data in enumerate(datas):
        if data[args.uniq_key] not in generate_ids2data:
            un_generated_datas.append(data)
        else: # resp 
            if len(generate_ids2data[data[args.uniq_key]]['responses']) < args.duplicate_num:
                un_generated_datas.append(data)

    if len(un_generated_datas) == 0:
        logger.info(f"all data has been generated")
        return [], len(datas)

    logger.info(f">> S{args.step_idx} cost time load data from file: {(time.time()-start_time)/60:.2f}min, total data: {len(datas)} - generated data: {len(generate_ids2data)}, ungen data: {len(un_generated_datas)}")

    # 4. split the ungenerated data into num of processes
    # un_generated_datas = split_data(args, un_generated_datas)

    # print(f"input_path: {args.dataset_path}\noutput_path: {args.output_path}")
    return un_generated_datas, len(datas) - len(un_generated_datas)


def sync_or_async_func_wrapper(func):
    if inspect.iscoroutinefunction(func):
     
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
    else:
        async def wrapper(*args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)
    return wrapper

# 测试 api 是否可用
def test_model(client, model_name):
    try:
        client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10,
        )
        return True
    except:
        return False
    

def extract_boxed_content(s):
    contents = []
    start = 0
    
    while True:
        start1 = s.find("\boxed{", start)
        start2 = s.find("\\boxed{", start)

        if start1 == -1 and start2 == -1:
            break
        elif start1 == -1 or (start1 > start2 and start2 != -1):
            start = start2
            lens = len("\\boxed{")
        else:
            start = start1
            lens = len("\boxed{")
        
        start += lens
        content = []
        stack = ['{']
        if start == len(s):
            break
        for i in range(start, len(s)):
            if s[i] == '{':
                stack.append('{')
            elif s[i] == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        break
            content.append(s[i])
        
        contents.append(''.join(content))  # Exclude the last closing brace
        start = i + 1  # Move past the current boxed content
    
    return contents

def parse_numbers_and_symbols(s):
    pattern = (
        r'-?\d+\.?\d*|'  # 
        r'\\pi|e|i|\\infty|π|∞|'  # 
        r'\\sqrt|\\log|\\sin|\\cos|\\tan|\\lim|\\sum|\\prod|\\Delta|'  # 
        r'\|x\||'  # 绝对值
        r'\\frac\{[^}]*\}\{[^}]*\}|'  # 
        r'\\nabla|\\forall|\\exists|\\in|\\subseteq|\\approx|\\equiv|\\neq|\\pm'  # 
    )
    matches = re.findall(pattern, s)
    return matches

def compare_strings(str1, str2):
    # 
    nums1 = parse_numbers_and_symbols(str1)
    nums2 = parse_numbers_and_symbols(str2)
    # 
    return nums1 == nums2

def parse_fraction(expr):
    expr = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', expr)
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', expr)
    return sympify(expr) if '/' in expr else Fraction(eval(expr))

def are_values_equal(str1, str2):
    # 
    str1 = str1.replace(', ', '').lower().strip().strip("$").strip()
    str2 = str2.replace(', ', '').lower().strip().strip("$").strip()
    if str1 == str2:
        return True
    elif 'rac' in str1 or 'rac' in str2:
        try:
            value1 = parse_fraction(str1)
            value2 = parse_fraction(str2)
            # 
            return value1 == value2
        except:
            return False
    else:
        try:
            return float(str1) == float(str2)
        except:
            return False
