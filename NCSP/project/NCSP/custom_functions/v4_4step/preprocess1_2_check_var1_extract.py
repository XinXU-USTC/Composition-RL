import random
import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data


PROMPT_CHECK = """{P1}

Assume that the final answer of the problem is {FINAL_ANSWER}. {DEFINITION_OF_NEW_VARIABLE1}

Then what is the value of $new_variable1$?

Please output the value of $new_variable1$ directly, wrapping it in \\boxed{}, for example, \\boxed{3}.

"""



# check modified p1
def main(datas: list) -> list:
    new_datas = []
    for data in datas:
        for idx, resp in enumerate(data['responses']):
            id = data['__id__']
            if idx > 0:
                id = f"{id}_{idx}"
            
            data1 = data['data1']
            data2 = data['data2']
            # Question,solution,Answer,subject,level,unique_id
            question1 = data1['problem']
            question2 = data2['problem']
            final_answer1 = data1['final_answer']
            final_answer2 = data2['final_answer']

            var_of_p1 = resp.split('<The_Value_of_New_Variable1>')[1].split('</The_Value_of_New_Variable1>')[0].strip()
            analysis1 = resp.split('<Analysis>')[1].split('</Analysis>')[0].strip()
            definition_of_var1 = resp.split('<The_Definition_of_New_Variable1>')[1].split('</The_Definition_of_New_Variable1>')[0].strip()
            modify_p1 = data['data1']['problem'] + definition_of_var1
            
            if any([var.lower() == 'none' for var in [var_of_p1, modify_p1, analysis1, definition_of_var1]]):
                continue

            prompt_step1 = PROMPT_CHECK.replace('{P1}', question1).replace('{FINAL_ANSWER}', final_answer1).replace('{DEFINITION_OF_NEW_VARIABLE1}', definition_of_var1)

            d1 = {
                '__id__': id,
                # 'prompt': f"<|user|>\n{prompt_step1}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt_step1}],
                'responses': [],
                'data1': data1,
                'data2': data2,
                'modify_p1': modify_p1,
                'var1': var_of_p1,
                'definition_of_var1': definition_of_var1,
                'analysis1': analysis1,
            }
            new_datas.append(d1)
        
    return new_datas


if __name__ == '__main__':
    path = '/code/infer_client/data/MATH7500_pair_gpt4o.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])