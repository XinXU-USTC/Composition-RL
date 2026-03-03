

def main(datas: list) -> list:
    new_datas = []
    template = """You are a math professor. Here are some math problems and corresponding answers. Please check the answers and give your feedback.
Problem: {problem}
Answer: {answer}

Analysis and output [[true/false]] in the last line.
"""
    for data in datas:
        for resp in data['responses']:
            msgs = data['messages']
            prompt = template.format(problem=data['query'], answer=resp)
            new_data = {'messages': [{'role':'user', 'content':prompt}], 'step1_resp': resp}
            new_datas.append(new_data)
    return new_datas
