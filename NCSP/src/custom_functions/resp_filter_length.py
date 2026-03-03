
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("checkpoint/THUDM/glm-4-9b-chat", trust_remote_code=True)

def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    return [resp for resp, finish_reason in zip(resps, finish_reasons) if resp and resp.strip() and (finish_reason is None or 'length' not in finish_reason)]

if __name__ == '__main__':
    data = {"prompt": "I am a good."}
    resps = ["I am a good."]
    finish_reasons = ["length"]
    main(data, resps, finish_reasons, [])