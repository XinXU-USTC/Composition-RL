


def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    
    return [resp for resp in resps if resp and resp.strip() and \
            '<IsContain>' in resp and '</IsContain>' in resp]
