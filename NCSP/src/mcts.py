# from math_evaluation import is_equiv 
import os
import traceback

from src.utils import get_function_from_file, sync_or_async_func_wrapper
os.environ["SYMPY_USE_CACHE"]="no"
os.environ["SYMPY_INT_TRACE"]="no"

import gc
import sympy as sp
from sympy.parsing.latex import parse_latex
import random
import requests
import tqdm.asyncio
import argparse
from openai import AsyncOpenAI, OpenAI
import multiprocessing as mp
from collections import deque
from multiprocessing import Process
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
import json
from datasets import load_dataset
import numpy as np
import warnings
from tqdm import tqdm
import time
import json
import random
from transformers import AutoTokenizer
from src.logging_config import logger_manager
logger = logger_manager(0)
args = None

warnings.simplefilter(action='ignore', category=FutureWarning)


async def gen_response(messages, clients, resp_server_names, configs):
    random_idx = random.randint(0, len(resp_server_names) - 1)
    server_name = resp_server_names[random_idx]
    client = clients[server_name]
    try:
            max_retry = args.max_try_per_request
            for retry in range(max_retry):
                try:
                    if type(messages) == list:
                        completion = await client.chat.completions.create(
                            model=server_name,
                            messages=messages,
                            stream=False,
                            extra_body=configs
                        )
                        if 'stop_reason' in completion.choices[0]:
                            stop_reasons = [choice.stop_reason for choice in completion.choices]
                        elif 'stop_reason' in completion.choices[0].model_extra:
                            stop_reasons = [choice.model_extra['stop_reason'] for choice in completion.choices]
                        else:
                            stop_reasons = [None] * len(completion.choices)
                        return [choice.message.content for choice in completion.choices], \
                            [choice.finish_reason for choice in completion.choices], \
                            stop_reasons
                        

                    elif  type(messages) == str:
                        completion = await client.completions.create(
                            model=server_name,
                            prompt=messages,
                            stream=False,
                            extra_body=configs
                        )
                        if 'stop_reason' in completion.choices[0]:
                            stop_reasons = [choice.stop_reason for choice in completion.choices]
                        elif 'stop_reason' in completion.choices[0].model_extra:
                            stop_reasons = [choice.model_extra['stop_reason'] for choice in completion.choices]
                        else:
                            stop_reasons = [None] * len(completion.choices)
                        return [choice.text for choice in completion.choices],\
                              [choice.finish_reason for choice in completion.choices], \
                              stop_reasons
                    else:
                        logger.error(f"generate error: messages type error {type(messages)}")
                        raise NotImplementedError
                except Exception as e:
                    logger.error(f"generate error: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error when gen resp: {e}")
        return [""], ["error"], ["error"]
    return [""], ["error"], ["error"]



class MCTS:
    def __init__(self, args_, clients):
        global args
        args = args_

        self.actor_clients = clients
        # self.T = args.T if hasattr(args, "T") else 128
        # self.cpuct = args.cpuct if hasattr(args, "cpuct") else 0.7
        # self.gamma = args.gamma if hasattr(args, "gamma") else 1.0
        # self.beta = args.beta if hasattr(args, "beta") else 1.0
        # self.eps = args.eps if hasattr(args, "eps") else 1e-8
        # self.threshold = args.threshold if hasattr(args, "threshold") else 0.0
        
        # funcs
        global preprocess_func_for_expand, parse_func_for_expand, preprocess_func_for_rollout, parse_func_for_rollout
        preprocess_func_for_expand = sync_or_async_func_wrapper(get_function_from_file(args.preprocess_func_for_expand, 'main'))
        parse_func_for_expand = sync_or_async_func_wrapper(get_function_from_file(args.parse_func_for_expand, 'main'))
        preprocess_func_for_rollout = sync_or_async_func_wrapper(get_function_from_file(args.preprocess_func_for_rollout, 'main'))
        parse_func_for_rollout = sync_or_async_func_wrapper(get_function_from_file(args.parse_func_for_rollout, 'main'))


    async def built_tree(self, data) -> bool:

        try:
            start_time = time.time()

            """
                1. 初始化根节点
            """
            # 1.1 Initialize the root node
            prompt, state, action, golden_answer = data['prompt'], data['state'], data.get('action', None), data.get('golden_answer', None)
            root = Node(prompt=prompt, golden_answer=golden_answer, state=state, action=action,
                            terminal=False, truncated=False,
                            actor_clients=self.actor_clients,
                            parent=None,  max_breadth = args.max_breadth_of_root, max_depth=args.max_depth, 
                            threshold=args.threshold, gamma=args.gamma, beta=args.beta, eps=args.eps)
            if args.debug: logger.debug(f"root init success: {time.time() - start_time}")
            # 1.2 Initialize Rollout for the root node
            # await node.rollout()

            """
                2. MCTS:
                    每次从根节点开始
                        1、如果当前节点expandable则expand + rollout + backup
                        2、否则selection下一节点
                    终止条件： expand总次数达到T

            """
            max_depth = 0
            for idx in range(args.T):
                # 2.1 Selection
                node = root
                while node and node.children and not node.expandable():
                    next_node = node.select() # 无孩子节点则返回None
                    node = next_node
                    max_depth = max(max_depth, node.depth)
                
                # 2.2 Expand + Rollout + Backup
                if node and node.expandable():
                    await node.expand()
                    if args.debug: logger.debug(f"{idx}-th node expand success: {time.time() - start_time:.2f}")
                    

            logger.info(f"cost: {time.time() - start_time}: nodes_num: {idx}, cur_depth: {max_depth}")
             
            # return root.get_preference()
            # root.save_to_json(writer=tree_writer)
            
            return [{'tree': root.to_dict(), 'best_trajectory': root.get_best_trajectory()}], ['success']
        
        except Exception as e:
            logger.error(f"PWP {traceback.format_exc()}")
            return [None],  [f'{e}']

class Node:
    def __init__(self, prompt, state, action, golden_answer, terminal, truncated, actor_clients, parent=None, max_breadth=3, max_depth=10, threshold=0.0, gamma=1.0, beta=1.0, eps=1e-8):
        self.timestamp = time.time()
        self.prompt = prompt
        self.golden_answer = golden_answer

        self.state = state
        self.action = action
        self.terminal = terminal
        self.actor_clients = actor_clients
        # self.critic_clients = critic_clients
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.q_values = {}
        self.depth = parent.depth + 1 if parent else 0
        self.R = 0
        self.correct = 0
        self.is_gt = False
        self._deadlooped = False
        self._used = {}
        self.rollout_result = None
        self.truncated = truncated
        # self.tokenizer = tokenizer

        # hyperparameters of backup and tree
        self.max_breadth = max_breadth
        self.gamma = gamma
        self.beta = beta
        self.max_depth = max_depth
        self.threshold=threshold
        self.eps = eps
    

    def expandable(self):
        # 1. 不是terminal 2. 没有达到最大深度 3. 孩子节点数小于最大宽度 4. 没有重复
        return not self.terminal and self.depth < self.max_depth and len(self.children.items()) < self.max_breadth and (not self._deadlooped)

    def select(self, ):
        """
            流程：
                1. 当前节点没有孩子节点，返回None
                2. 有孩子节点，则计算每个孩子节点的puct值找出最大的那个，多个最大的话随机选一个
                    puct = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
                
        """
        if not self.children:
            return None
        best_value = -float('inf')
        best_node = []
        # probs = [self.q_values.get(action, 0) + cpuct * np.sqrt(self.visits) / (child.visits + 1) for action, child in self.children.items()]
        for action, child in self.children.items():
            puct = self.q_values.get(action, 0) + args.cpuct * np.sqrt(self.visits) / (child.visits + 1)
            if puct > best_value:
                best_value = puct
                best_node = [child]
            elif puct == best_value:
                best_node.append(child)
        return random.sample(best_node, 1)[0]

    async def expand(self):
        # 1. get the action from the model
        if len(self.children.items()) >= self.max_breadth or self.depth >= self.max_depth:
            await self.rollout()
            return
        # 1.1 CUSTOM FUNC 预处理
        prompt = await preprocess_func_for_expand(args, self)
        if args.debug: logger.debug(f"preprocess for expand success: prompt = {prompt}")

        # 1.2 生成response
        responses, finish_reasons, stop_reasons = await gen_response(
            # messages=[{"role": "user", "content": prompt}],
            messages=prompt,
            clients=self.actor_clients,
            resp_server_names=args.resp_server_names,
            configs = args.expand_config
        )
        if args.debug: logger.debug(f"gen expand resp success: resp = {responses}")
        # 1.3 CUSTOM FUNC 解析response
        action, new_state, terminal, truncated = await parse_func_for_expand(args, self, responses, finish_reasons, stop_reasons)
        if args.debug: logger.debug(f"parse expand resp success:  action = {action}, new_state = {new_state}, terminal = {terminal}, truncated = {truncated}")

        # 2. expand new node
        # 2.1 rollout if the action is not used
        if not action in self._used.keys():
            self._used[action] = 1
            new_node = Node(prompt=self.prompt, state=new_state, action=action,
                            golden_answer=self.golden_answer, terminal=terminal, truncated=truncated,
                            actor_clients=self.actor_clients,
                            parent=self, max_breadth=args.max_breadth_of_children, max_depth=self.max_depth,
                            threshold=self.threshold, gamma=self.gamma, beta=self.beta, eps=self.eps)
            self.children[action] = new_node
            await new_node.rollout()
            if args.debug: logger.debug(f"rollout for new node success: {time.time() - self.timestamp:.2f}")

        else:
            self._used[action] += 1
        # 2.2 check if _deadlooped
        if self._used[action] == self.max_breadth:
            self._deadlooped = True

    async def rollout(self):
        # prompt = INSTRUCTION.format(self.prompt, self.state)
        prompt = await preprocess_func_for_rollout(args, self)
        if args.debug: logger.debug(f"preprocess for rollout success: prompt = {prompt}")
        responses, finish_reasons, stop_reasons = await gen_response(
            # messages=[{"role": "user", "content": prompt}],
            messages=prompt,
            clients=self.actor_clients,
            resp_server_names=args.resp_server_names,
            configs = args.rollout_config
        )
        if args.debug: logger.debug(f"gen rollout resp success: resp = {responses}")
        self.correct, self.R, self.rollout_result  = await parse_func_for_rollout(args, self, responses, finish_reasons, stop_reasons)
        if args.debug: logger.debug(f"parse rollout resp success: correct = {self.correct}, R = {self.R}, rollout_result = {self.rollout_result}")
        self.backup()
        return self.R

    # TODO: check
    def backup(self):
        self.visits += 1
        true_count, false_count = 0, 0
        if self.children:
            nume, deno = 0, 0
            for action, child in self.children.items():
                reward = child.R - self.R if not self.is_gt else 0
                self.q_values[action] = self.beta * reward + self.gamma * child.value
                nume += self.q_values[action] * child.visits
                deno += child.visits
                true_count += child.correct > 0
                false_count += child.correct < 0
            if nume and deno:
                self.value = nume / deno
        else:
            self.value = self.R
        if self.children:
            try:
                self.correct = 1 if (true_count and 1.0 * true_count / (true_count + false_count) >= self.threshold - args.eps) else -1
            except:
                self.correct = 0 # in case all decision are unclear

        if self.parent is not None:
            self.parent.backup()

    def to_dict(self):
        """递归将当前节点及其子节点转换为字典"""
        return {
            'prompt': self.prompt,
            'state': self.state,
            'action': self.action,
            'golden_answer': self.golden_answer,
            'terminal': self.terminal,
            'truncated': self.truncated,
            # 'actor_clients': self.actor_clients,
            # 'critic_clients': self.critic_clients,
            'rollout_result': self.rollout_result,
            'visits': self.visits,
            'value': self.value,
            'q_values': self.q_values,
            'depth': self.depth,
            'R': self.R,
            'b1': self.b1,
            'correct': self.correct,
            'b2': self.b2,
            'gamma': self.gamma,
            'is_gt': self.is_gt,
            'beta': self.beta,
            'timestamp': self.timestamp,
            'threshold': self.threshold,
            'max_depth': self.max_depth,
            'children': {key: child.to_dict() for key, child in self.children.items()}
        }

    def save_to_json(self, writer):
        """将当前节点及其子节点的树结构保存为JSON dict"""
        tree_dict = self.to_dict()
        writer.write(json.dumps(tree_dict, ensure_ascii=False) + '\n')
        return tree_dict

    def get_best_trajectory(self,):
        best_trajectory = []
        node = self
        while node:
            best_trajectory.append({key: val for key, val in node.__dict__.items() if key in [
                'timestamp', 'prompt', 'state', 'action', 'terminal', 'visits', 'rollout_result', 'value', 'q_values',
                  'depth', 'R', 'correct', 'is_gt', 'truncated']})
            node = node.select()

        logger.info(f"node has steps {len(best_trajectory)}")
        return best_trajectory

