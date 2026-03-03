from math_evaluation import is_equiv 
import os
os.environ["SYMPY_USE_CACHE"]="no"
os.environ["SYMPY_INT_TRACE"]="no"

import gc
import sympy as sp
from sympy.parsing.latex import parse_latex
import random
import requests
import tqdm.asyncio
import argparse
from openai import OpenAI
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

# ACTOR_PORT_LIST = [8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015]
# CRITIC_PORT_LIST = [8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015]

ACTOR_PORT_LIST = [8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015]
CRITIC_PORT_LIST = [8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015]


actor_name = os.environ.get("MODEL_NAME", "OpenRLHF/ckpt/mpo-glm-128R-depth-0729-woabs-iter1-beta1-valuegap1/")
critic_name = os.environ.get("MODEL_NAME", "OpenRLHF/ckpt/mpo-glm-128R-depth-0729-woabs-iter1-beta1-valuegap1/")

NUM_THREADS = 10
NUM_PROCESS = mp.cpu_count()

# NUM_THREADS = 2
# NUM_PROCESS = 2

eps = 1e-6
CONTEXT_LIMIT = 2048
QUEUE_SIZE = 16384

warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_TOKENS = 896
additional_length = 128

sep_token = "\n\n"
actor_stop_token = "<|endoftext|>" # for completion model
stop_token_ids = ['151329', '151336', '151338']
# actor_stop_token = "<|user|>" # for chat model
critic_stop_token = "<|user|>" # for chat model

INSTRUCTION = """<|user|>\n{}<|assistant|>\n{}"""

EXTRACT_PROMPT = """
Given the following solution, extract the final answer and present it in a clear and concise manner.
Note that you should always wrap the answer in <Answer></Answer>. If the solution is not complete, you should complete it but not give out the solution. You should only output the answer wrapped in <Answer></Answer>. No other information.

# Example
## Given Solution
To determine if the expressions are equivalent, we simplify each one:
1. Simplify \( \frac{{2x + 4}}{{2}} \):
   \[ \frac{{2x + 4}}{{2}} = \frac{{2x}}{{2}} + \frac{{4}}{{2}} = x + 2 \]
2. Compare the simplified expression with \( x + 2 \):
   \[ x + 2 \equiv x + 2 \]
Since both expressions are equal, the final answer is **Yes**.
## Extracted Answer
<Answer>Yes</Answer>

# Example
## Given Solution
The value of 123 * 45 + 16 can be calculated step by step.
123 * 45 = (100 + 20 + 3) * (40 + 5) + 16 = 4000 + 800 + 120 + 500 + 100 + 15 + 16 = 4800 + 620 + 131 = 4800 + 751 = 5551. So the final solution is 5551.
## Extracted Answer
<Answer>5551</Answer>

# Task
## Given Solution
{}
## Extracted Answer
"""

SIMPLIFY_PROMPT = """
I will give you a math problem answer, you should simplify it according to some principles into a pure math expression.
- Remove any units. If the answer is given as a equation with several units, then always keep the first one. 3 purple balls
- Keep the result clean. If the answer has special tokens like $ or delimiters, then remove them. The clean principles is keep the result like latex format.
- Remove redundant signals. For example, the parenthesis are not useful so we move them.
- If the answer is using double slash, then use just one.

Do not give any explanation, just give the simplified answer.

# Examples

## Raw Answer
3 purple balls
## Simplified Answer
3

## Raw Answer
\(\frac{{1}}{{2}}\)
## Simplified Answer
\frac{{1}}{{2}}.

## Raw Answer
\\frac{{1}}{{2}} 
## Simplified Answer
\frac{{1}}{{2}}

# Task
## Raw Answer
{}
## Simplified Answer
"""

EQUATION_PROMPT = """
Given part of a solution, determine whether it contains any math or numerical expression . Give the final judgement wrapped in <Judge></Judge>.
Note that any expression related math can be viewed as a math expression, your task is to distinguish them from pure text solution
**If the part of solution contains at least one math expression, judge them as <Judge>T</Judge> meaning True, otherwise <Judge>F</Judge> as False.** You should only output the answer wrapped in <Judge></Judge>. No other information.
Do not add any additional solution.

# Example
## Solution
To solve the problem, we notice that \dfrac{{2x+4}}{{2}} can be deduced into x+2.
## Response
<Judge>T</Judge>

# Example
## Expression1
To answer the question, we have to do some calculation:
## Response
<Judge>F<Judge>

# Example
## Expression1
The answer to the question is 10. We got it!
## Response
<Judge>T<Judge>


# Task
## Solution
{}
## Response

"""

CHECK_PROMPT = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications.

Examples:

    Expression 1: 21
    Expression 2: 21 canoes

Yes

    Expression 1: 59,049
    Expression 2: 59049

Yes

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

# Judge whether the *completion model* terminates
def is_terminal(action, finish_reason, stop_reason):
    terminal = "stop" in finish_reason and (stop_reason == None or stop_reason == actor_stop_token or stop_reason == critic_stop_token or str(stop_reason) in stop_token_ids)
    truncated = "len" in finish_reason
    return terminal, truncated

def check_correctness(question, solution, golden_answer, model, tokenizer):
    max_retry = 5
    response_tokens = 100 # considering the answer may be an expression
    delta_tokens = 50

    for retry in range(max_retry):
        try:
            # check_logger.critical(f"Retry: {retry}, Solution: {repr(solution)}")
            prompt = EXTRACT_PROMPT.format(solution)
            while len(tokenizer.encode(prompt)) > CONTEXT_LIMIT - response_tokens - delta_tokens:
                solution = solution[min(int(len(solution) / 2), 50): ]
                prompt = EXTRACT_PROMPT.format("..." + solution)

            my_answer, _, _ = gen_response( 
                messages=[{"role": "user", "content": prompt}],
                clients=model,
                n=1,
                max_tokens=response_tokens, # for answer
                stop_at_sep=False,
                temperature=0,
                top_p=1,
                mode="chat"
            )

            my_answer = my_answer[0].split("<Answer>")[-1].split("</Answer>")[0]
            if my_answer == "":
                continue
            my_answer, _, _ = gen_response( 
                messages=[{"role": "user", "content": SIMPLIFY_PROMPT.format(my_answer)}],
                clients=model,
                n=1,
                max_tokens=response_tokens, # for answer
                stop_at_sep=False,
                temperature=0,
                top_p=1,
                mode="chat"
            )

            my_answer = my_answer[0].strip()
            if my_answer == "":
                continue
            prompt = CHECK_PROMPT % {"expression1": golden_answer, "expression2": my_answer}
            try:
                # TODO: fix memory leakage
                raise NotImplementedError
                result = 1 if is_equiv(golden_answer, my_answer) else -1
            except Exception as e:
                output, _, _ = gen_response( 
                    messages=[{"role": "user", "content": prompt}],
                    clients=model,
                    n=1,
                    max_tokens=response_tokens,
                    stop_at_sep=False,
                    temperature=0,
                    top_p=1,
                    mode="chat"
                )
                output = output[0].lower().strip()
                if output == "yes":
                    result = 1
                elif output == "no":
                    result = -1
                else:
                    continue
                if not result:
                    result = 0
            # print(my_answer, golden_answer, result)
            return result, my_answer

        except Exception as e:
            print(f'{retry}:, {e}')
            pass
    return 0, "<UNK>"

def gen_response(messages, clients, n=1, max_tokens=MAX_TOKENS, stop_at_sep=False, temperature=0.9, top_p=1.0, mode="completion", use_webapi=False):
    try:
        if not use_webapi:
            stop = [sep_token] if stop_at_sep else []
            max_retry = 5
            for retry in range(max_retry):
                try:
                    if mode == "chat":
                        stop.append(critic_stop_token)
                        idx = random.randint(0, len(clients) - 1)
                        completion = clients[idx].chat.completions.create(
                            model=critic_name,
                            temperature=temperature,
                            top_p=top_p,
                            messages=messages,
                            n=n,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                        return [choice.message.content for choice in completion.choices], [choice.finish_reason for choice in completion.choices], [choice.stop_reason for choice in completion.choices]

                    else:
                        stop.append(actor_stop_token)
                        idx = random.randint(0, len(clients) - 1)
                        completion = clients[idx].completions.create(
                            model=actor_name,
                            temperature=temperature,
                            top_p=top_p,
                            prompt=messages,
                            n=n,
                            max_tokens=max_tokens,
                            stop=stop
                        )
                        return [choice.text for choice in completion.choices], [choice.finish_reason for choice in completion.choices], [choice.stop_reason for choice in completion.choices]
                except Exception as e:
                    print(f"generate error: {e}")
                    continue
        else:
            max_retry = 3
            url = "http://172.21.64.38:8080/generate" # glm-4-public
            headers = { "Content-Type": "application/json" }
            if stop_at_sep:
                stop = [sep_token]
            else:
                stop = []
            # TODO: diffentiate chat / completion
            data = {
                "stream": False,
                "inputs": messages[0]["content"],
                "parameters": {
                    "best_of": 1,
                    "decoder_input_details": False,
                    "details": False,
                    "do_sample": True,
                    "max_new_tokens": max_tokens,
                    "temperature":temperature,
                    "top_p": top_p,
                    "stop": stop
                }
            }
            
            responses = []
            for _ in range(n):
                flag = 0
                for _ in range(max_retry):
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    if response.status_code == 200:
                        raw_responses = response.json()
                        responses.append(raw_responses["generated_text"])
                        flag = 1
                        break
                    else:
                        print("Failed to get response. Status code:", response.status_code)
                        print("Response body:", response.text)
                if flag == 0:
                    responses.append("API ERROR")

            return responses, _, _ 
    except Exception as e:
        print(f"Error when gen resp: {e}")
        return [""], ["error"], ["error"]
    return [""], ["error"], ["error"]

class MCTS:
    def __init__(self, actor_clients, critic_clients, cpuct=0.7, gamma=1.0, T=128, b1=3, b2=3):
        self.b1 = b1
        self.b2 = b2
        self.actor_clients = actor_clients
        self.critic_clients = critic_clients
        self.cpuct = cpuct
        self.gamma = gamma
        self.T = T

    def search(self, threshold, prompt, golden_answer=None, solution=None):
        try:
            tokenizer = AutoTokenizer.from_pretrained(critic_name, trust_remote_code=True) # for judgement, except for exceeding the max tokens
            start_time = time.time()
            if solution:
                root = Node(prompt=prompt, state=solution, action=solution,
                            golden_answer=golden_answer, terminal=False, truncated=False,
                            actor_clients=self.actor_clients,
                            critic_clients=self.critic_clients,
                            parent=None, b1=self.b1, b2=self.b2, 
                            tokenizer=tokenizer, threshold=threshold)
            else:
                root = Node(prompt=prompt, state="", action="",
                            golden_answer=golden_answer, terminal=False, truncated=False,
                            actor_clients=self.actor_clients, 
                            critic_clients=self.critic_clients,
                            parent=None, b1=self.b1, b2=self.b2, 
                            tokenizer=tokenizer, threshold=threshold)
            node = root
            node.rollout()
            
            for idx in range(self.T):
                print(f"Total: {self.T}, Now at step: {idx}, cost: {time.time() - start_time}; Father depth: {node.depth}")
                flag = 1
                node = root
                while node.children:
                    max_breadth = node.b1 if node.depth == 0 else node.b2
                    current_depth = node.depth
                    if not node.terminal and current_depth < node.max_depth and len(node.children.items()) < max_breadth and (not node._repeated):
                        node.expand()
                        flag = 0
                        break
                    next_node = node.select(self.cpuct)
                    if next_node:
                        node = next_node
                    else:
                        break
                max_breadth = node.b1 if node.depth == 0 else node.b2
                current_depth = node.depth
                if flag and not node.terminal and current_depth < node.max_depth and len(node.children.items()) < max_breadth and (not node._repeated):
                    node.expand()
                elif node.terminal:
                    node.backup()
            return root.get_preference()
        except Exception as e:
            print("PWP", e.with_traceback())
            raise e

def get_interest_attributes(node, is_cur_node=True):
    res = {
        "timestamp": node.timestamp,
        "prompt": node.prompt,
        "state": node.state,
        "action": node.action,
        "terminal": node.terminal,
        "visits": node.visits,
        "rollout_result": node.rollout_result,
        "value": node.value,
        "q_values": node.q_values,
        "depth": node.depth,
        "R": node.R,
        "correct": node.correct,
        "is_gt": node.is_gt,
        "truncated": node.truncated
    }
    if is_cur_node:
        if node.children:
            res["children"] = [get_interest_attributes(child, is_cur_node=False) for child in node.children.values()],
        res["parent"] = get_interest_attributes(node.parent, is_cur_node=False) if node.parent else None
    return res

class Node:
    def __init__(self, prompt, state, action, golden_answer, terminal, truncated, actor_clients, critic_clients, parent=None, b1=3, b2=3, tokenizer=None, max_depth=10, threshold=0.0):
        self.threshold=threshold
        self.timestamp = time.time()
        self.prompt = prompt
        self.state = state
        self.action = action
        self.max_depth = max_depth
        self.terminal = terminal
        self.truncated = truncated
        self.golden_answer = golden_answer
        self.actor_clients = actor_clients
        self.critic_clients = critic_clients
        self.parent = parent
        self.children = {}
        self.rollout_result = "<UNK>"
        self.visits = 0
        self.value = 0
        self.q_values = {}
        self.depth = parent.depth + 1 if parent else 0
        self.R = 0
        self.b1 = b1
        self.correct = 0
        self.b2 = b2
        self.expand_count = self.b1 if self.depth == 0 else self.b2
        self.gamma = 1.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_gt = False
        self._repeated = False
        self._used = {}
        self.tokenizer = tokenizer
        self.beta = 0.3
        
    def select(self, cpuct):
        if not self.children:
            return None
        best_value = -float('inf')
        best_node = []
        # probs = [self.q_values.get(action, 0) + cpuct * np.sqrt(self.visits) / (child.visits + 1) for action, child in self.children.items()]
        for action, child in self.children.items():
            puct = self.q_values.get(action, 0) + cpuct * np.sqrt(self.visits) / (child.visits + 1)
            if puct > best_value:
                best_value = puct
                best_node = [child]
            elif puct == best_value:
                best_node.append(child)
        return random.sample(best_node, 1)[0]

    def expand(self):
        max_breadth = self.b1 if self.depth == 0 else self.b2
        current_depth = self.depth
        if len(self.children.items()) >= max_breadth or current_depth >= self.max_depth:
            self.rollout()
            return
        prompt = INSTRUCTION.format(self.prompt, self.state)
        if self.state == "":
            prompt = prompt.replace("\n\n", "")
        responses, finish_reasons, stop_reasons = gen_response(
            # messages=[{"role": "user", "content": prompt}],
            messages=prompt,
            clients=self.actor_clients,
            n=1,
            stop_at_sep=True,
            max_tokens=additional_length,
            temperature=0.99,
            top_p=1,
            mode="completion"
        )

        action, finish_reason, stop_reason = responses[0].strip(), finish_reasons[0], stop_reasons[0]
        new_state = f"{self.state}{action} \n\n "
        
        terminal, truncated = is_terminal(action, finish_reason, stop_reason)
        new_node = Node(prompt=self.prompt, state=new_state, action=action,
                        golden_answer=self.golden_answer, terminal=terminal, truncated=truncated,
                        actor_clients=self.actor_clients, 
                        critic_clients=self.critic_clients, 
                        parent=self, b1=self.b1, b2=self.b2, 
                        tokenizer=self.tokenizer, threshold=self.threshold)
        if not action in self._used.keys():
            self._used[action] = 1
            self.children[action] = new_node
            new_node.rollout()
        else:
            self._used[action] += 1
        if self._used[action] == max_breadth:
            self._repeated = True
            return

    def rollout(self):
        prompt = INSTRUCTION.format(self.prompt, self.state)
        responses, _, _ = gen_response(
            # messages=[{"role": "user", "content": prompt}],
            messages=prompt,
            clients=self.actor_clients,
            n=1,
            max_tokens=None,
            stop_at_sep=False,
            temperature=0.1,
            top_p=1,
            mode="completion"
        )
        correctness_reward, rollout_result = check_correctness(self.prompt, self.state + responses[0], self.golden_answer, self.critic_clients, self.tokenizer)
        self.correct = correctness_reward
        self.R = correctness_reward
        # self.R = correctness_reward + 0.1 * self.R # TODO: hyperparam
        self.rollout_result = rollout_result
        self.backup()
        return self.R

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
                self.correct = 1 if (true_count and 1.0 * true_count / (true_count + false_count) >= self.threshold - eps) else -1
            except:
                self.correct = 0 # in case all decision are unclear

        if self.parent is not None:
            self.parent.backup()

    def get_preference(self):
        items = {}
        items["prompt"] = self.prompt
        items["golden_answer"] = self.golden_answer
        items["nodes"] = []
        items["rollout_result"] = self.rollout_result
        queue = deque()
        queue.append(self)
        count = 0
        while queue:
            count += 1
            node = queue.popleft()
            items["nodes"].append(get_interest_attributes(node))
            if node.children:
                for action, child in node.children.items():
                    queue.append(child)
        print("node has steps", count)
        return items

def get_prompts(dataset_name, is_math=False, is_test=False):
    """
    In the format of [(question, golden_answer)]
    """
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")["train"]
        prompts = [(data["question"], data["answer"].split("####")[-1].strip()) for data in dataset]
    elif dataset_name == "MATH":
        dataset = load_dataset("json", data_files="MATH-data.json")["train"]
        prompts = [(data["question"], data["answer"]) for data in dataset]
    elif dataset_name == "microsoft/orca-math-word-problems-200k":
        dataset = load_dataset("json", data_files="orca-data.json")["train"]
        prompts = [(data["question"], data["answer"]) for data in dataset]
    elif dataset_name == "TIGER-Lab/MATH-plus":
        dataset = load_dataset("json", data_files="mathplus-data.json")["train"]
        prompts = [(data["question"], data["answer"]) for data in dataset]
    elif dataset_name.endswith("json") or dataset_name.endswith("jsonl"):
        if dataset_name.endswith("json"):
            dataset = load_dataset("json", data_files=dataset_name)["train"]
        else:
            with open(dataset_name, "r") as f:
                dataset = f.readlines()
            dataset = [json.loads(line) for line in dataset]
        with open("Priority-Based-Tree-Search/results/tree/glm-chat-0729-checker-iter2-woabs-valuegap1.json", "r") as f:
            origin = [json.loads(line) for line in f.readlines()]
        prompts = [ori["prompt"] for ori in origin]
        dataset = [data for data in dataset if not data["question"] in prompts]
        print(f'len prompts: {len(prompts)}, len dataset: {len(dataset)}')
        questions = [data["question"] for data in dataset]
        if is_test:
            return dataset
            solutions = [[piece["content"] for piece in data["solution"].values() if piece] for data in dataset]
            labels = [[piece["label"] for piece in data["solution"].values() if piece] for data in dataset]
            if is_math:
                golden_answers = [data["golden_answer"] for data in dataset]
                prompts = [(questions[i], golden_answers[i], solutions[i], labels[i]) for i in range(len(questions))]
            else:
                prompts = [(questions[i], solutions[i]) for i in range(len(questions))]
        else:
            if is_math:
                golden_answers = [data["golden_answer"] for data in dataset]
                prompts = [(questions[i], golden_answers[i]) for i in range(len(questions))]
            else:
                prompts = [(questions[i],) for i in range(len(questions))]
    else:
        raise NotImplementedError
    return prompts

def handle_result(future, done_queue):
    response = future.result()
    if response is not None:
        done_queue.put(json.dumps(response))
    del response

def worker_build_training_pair(task_queue, done_queue, worker_func, count, threshold):
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:        
        futures = []
        for line in iter(task_queue.get, "STOP"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip malformed JSON
            if count == 1:
                future = executor.submit(worker_func, threshold, item[0])  # problem
            elif count == 2:
                future = executor.submit(worker_func, threshold, item[0], item[1])  # problem, golden_answer / solution
            elif count == 3:
                future = executor.submit(worker_func, threshold, item[0], item[1], item[2])  # problem, golden_answer, solution
            future.add_done_callback(lambda f: handle_result(f, done_queue))
            futures.append(future)
        
        for future in futures:
            try:
                future.result()
            except:
                continue
        done_queue.put("COMPLETE")
        
def build_training_file(dataset_name, output_file, is_test=False, is_math=False, threshold=0.0):
    num_processes = NUM_PROCESS
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)
    cnt = 0
    prompts = get_prompts(dataset_name, is_test=is_test, is_math=is_math)
    if is_test:
        for data in tqdm(prompts):
            state = ""
            golden_answer = data["golden_answer"]
            question = data["question"]
            for idx, step in data["solution"].items():
                if idx == "Step 1" and step["label"] == -1:
                    break
                if not step:
                    continue
                state = f'{state}{step["content"]} \n\n'
                task_queue.put(json.dumps((question, golden_answer, state)))
                cnt += 1
                if step["label"] == -1:
                    break
    else: 
        for prompt in prompts:
            task_queue.put(json.dumps(prompt))
            cnt += 1

    print("Read files done:", cnt)

    for _ in range(num_processes):
        task_queue.put('STOP')
    processes = []

    actor_clients, critic_clients = [], []
    for port in ACTOR_PORT_LIST:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="test")
        if test_model(client, actor_name):
            actor_clients.append(client)
    for port in CRITIC_PORT_LIST:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="test")
        if test_model(client, critic_name):
            critic_clients.append(client)
    mcts = MCTS(actor_clients, critic_clients)

    for _ in range(num_processes):
        count = 1 + is_test + is_math
        process = Process(target=worker_build_training_pair, args=(task_queue, done_queue, mcts.search, count, threshold))
        process.start()
        processes.append(process)
    progress_bar = tqdm()
    print("----- GOGOGOGOGOGOGO !!!!!")
    with open(output_file, 'w') as w:
        num_finished = 0
        num_save = 0
        while num_finished < num_processes:
            item = done_queue.get()
            if item == None:
                continue
            elif item == 'COMPLETE':
                num_finished += 1
            else:
                w.write(json.dumps(json.loads(item), ensure_ascii=False) + '\n')
                w.flush()
                num_save += 1
                print(f'save {num_save} examples to {output_file}', end='\r')
                progress_bar.update(1)
    progress_bar.close()
    for process in processes:
        process.join()
    print("All processes have finished")
    
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

if __name__ == "__main__":   
    # mp.set_start_method('fork')
    assert actor_name != "OpenRLHF/ckpt/mpo-glm-128R-depth-0729-woabs-iter1-beta1-valuegap1/"
    time.sleep(90)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset_name", type=str, default="./datasets/selected_500_len_6.json")
    parser.add_argument("--output_name", type=str, default="./results/tree/true_mcts_selected_500_len_6.json")
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--is_math", action="store_true")
    parser.add_argument("--use_webapi", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    output_name = args.output_name
    is_test = args.is_test
    is_math = args.is_math
    use_webapi = args.use_webapi
    threshold = args.threshold
    build_training_file(dataset_name, output_name, is_test, is_math, threshold)