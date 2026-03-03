

def main(data: dict, resps: list, finish_reasons:list, stop_reasons:list) -> bool:
    new_resps = []
    for resp, finish_reason in zip(resps, finish_reasons):
        if resp and resp.strip() and 'length' not in finish_reason:
            if "\boxed{" in resp or "\\boxed{" in resp:
                new_resps.append(resp)
    return new_resps
            

