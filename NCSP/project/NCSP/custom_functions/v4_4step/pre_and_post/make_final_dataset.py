import sys
import time
sys.path.append('.')

from copy import deepcopy

from src.utils import are_values_equal, extract_boxed_content, get_hash, load_data, save_data



import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)

PROBLEM_TEMPLATE = """{modify_p1}
{relation}
{modify_p2}"""


SOLUTION_TEMPLATE = """{s1}
Therefore, the $new_variable1$ is {val_of_p1}.

{calculation}

{s2}"""


def main(args, is_save = True) -> list:
    
    new_datas = []
    path = args.path
    save_path = args.save_path

    datas = load_data(path)

    var2_equal = 0
    same_var = 0
    for data in datas:
        # Question,solution,Answer,subject,level,unique_id
        resp = data['responses'][0]
        
        is_correct = resp.split('<IsContain>')[1].split('</IsContain>')[0].strip()
        if is_correct.lower() == 'yes':
            same_var += 1

        symbol_of_var1 = data['symbol_of_var1']
        symbol_of_var2 = data['symbol_of_var2']
        
        # 1. Get Question
        modify_p1 = data['modify_p1']
        final_problem = PROBLEM_TEMPLATE.format(modify_p1=modify_p1, relation=data['relationship'], modify_p2=data['modify_p2'])
        final_problem = final_problem.replace('new_variable1', symbol_of_var1).replace('new_variable2', symbol_of_var2)

        var2_by_calculation = data['var2_by_calculation']
        # check variable2
        if are_values_equal(data['var2'].lower(), var2_by_calculation.lower()):
            var2_equal += 1

        # 2. Get solution， s1 + relationship_calculation + s2
        s1 = data['data1']['solution'].split('[RESULT]')[0].strip()
        s1 = s1.replace('\\boxed', '')
        final_solution = SOLUTION_TEMPLATE.format(s1=s1, val_of_p1=data['var1'], calculation=data['output_of_s1s2'], val_of_p2=data['var2'], s2=data['data2']['solution'])     
        final_solution = final_solution.replace('new_variable1', symbol_of_var1).replace('new_variable2', symbol_of_var2)

        d = {
            '__id__': data['__id__'],
            'messages': [{'role': 'user', 'content': final_problem}, {'role': 'assistant', 'content': final_solution}],
            'data1': data['data1'],
            'data2': data['data2'],

            'modify_p1': data['modify_p1'],
            'var1': data['var1'],
            'definition_of_var1': data['definition_of_var1'],
            'analysis1': data['analysis1'],
            
            'modify_p2': data['modify_p2'],
            'var2': data['var2'],
            'definition_of_var2': data['definition_of_var2'],
            'analysis2': data['analysis2'],

            'code_of_p1p2': data['code_of_p1p2'],
            'output_of_p1p2': data['output_of_p1p2'],
            'relationship': data['relationship'],
            'difference': data['difference'],

            'code_of_s1s2': data['code_of_s1s2'],
            'output_of_s1s2': data['output_of_s1s2'],
            'var2_by_calculation': data['var2_by_calculation'],

            'symbol_of_var1': symbol_of_var1,
            'symbol_of_var2': symbol_of_var2,

            'final_answer': data['data2']['final_answer'],
        }
        new_datas.append(d)
    if is_save:
        save_data(new_datas, save_path)
    print(f"var2_equal: {var2_equal}, {var2_equal*100/len(new_datas):.2f}%")
    return new_datas

if __name__ == '__main__':
    args = parser.parse_args()
    new_datas = main(args)
    print(len(new_datas))
    print(new_datas[0])