
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./checkpoint/THUDM/glm-4-9b-chat", trust_remote_code=True)

def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    return_resps = []
    for resp, finish_reason in zip(resps, finish_reasons) :
        if resp and resp.strip():
            if (finish_reason is None or 'length' not in finish_reason):
                return_resps.append(resp)
            else:
                new_lens = 8000-len(tokenizer.encode(data['prompt']))
                data['generate_config'] = {
                        'seed': 42,
                        'temperature': 0.7,
                        'top_p': 0.95,
                        'max_new_tokens': new_lens,
                        'details': True,
                        'decoder_input_details': False,
                        'stop': ["<|endoftext|>", "<|user|>", "<|observation|>"],
                        'eos_token_id': [151329, 151336, 151338],
                        'pad_token_id': 151329
                    }
                print(f"Reconfig max_new_tokens to {new_lens}")
    return return_resps

if __name__ == '__main__':
    data = {"prompt": "I am a good."}
    resps = ["I am a good."]
    finish_reasons = ["length"]
    main(data, resps, finish_reasons, [])