


import json


def main(datas: list) -> list:
    path = 'datasets/PersonaHub/math.jsonl'
    datas2 = []
    with open(path, 'r') as f:
        datas2 = f.readlines()
    datas2 = [json.loads(data) for data in datas2]

    querys1 = [data['query'] for data in datas]

    new_datas = []
    for data2 in datas2:
        query = data2['query']
        if query not in querys1:
            new_datas.append(data2)
            
    return new_datas
