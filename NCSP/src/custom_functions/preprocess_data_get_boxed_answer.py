from copy import deepcopy


def main(datas: list) -> list:
    template = """Given the following math problem and answer, please extract the final answer. The final answer is a number or a formula.

Problem: {problem}
Answer: {answer}

Directly extract the final answer and do not output any other information:

"""
    new_datas = []
    for data in datas:
        for resp in data['responses']:
            msgs = data['messages']
            prompt = template.format(problem=data['query'], answer=resp)
            new_data = deepcopy(data)
            new_data['step1_resp'] = resp
            new_data['messages'] = [{'role': 'user', 'content': prompt}]
            new_datas.append(new_data)
    return new_datas
