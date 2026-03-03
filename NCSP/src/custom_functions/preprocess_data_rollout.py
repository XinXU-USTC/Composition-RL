


def main(datas: list) -> list:
    split_token = "\n"
    new_datas = []
    for data in datas:

        inputs = f"<|user|>\n{data['prompt']}<|assistant|>\n"
        response_pieces = data['response'].split(split_token)
        response_pieces = [split_token.join(response_pieces[:i+1]) for i in range(len(response_pieces))]
        for response_piece in response_pieces:
            new_data = {'prompt': inputs+response_piece, 'source': data['source'], 'id': data['id'], 'response': response_piece}
            new_datas.append(new_data)
    return new_datas
