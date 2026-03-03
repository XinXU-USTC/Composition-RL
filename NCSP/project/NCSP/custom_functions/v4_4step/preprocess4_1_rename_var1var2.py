import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data


# var1 va2 rename
PROMPT = """Given a math problem, your task is to find a new variable name that never appears in the problem for $new_variable1$ and $new_variable2$.
Note that:
1. The new variable names should be different from all the variables in the problem.
2. The new variable names should wraped in $$, for example, $m$, $\alpha$.

---

### Output Format:
<The_Symbol_of_New_Variable1>
(the new variable name for $new_variable1$, no other text)
</The_Symbol_of_New_Variable1>

<The_Symbol_of_New_Variable2>
(the new variable name for $new_variable2$, no other text)
</The_Symbol_of_New_Variable2>

---

### Example:
#### Problem:
A number is divisible by $9$ if the sum of its digits is divisible by $9.$ For example, the number $19\\,836$ is divisible by $9$ but $19\\,825$ is not.

If $D\\,767\\,E89$ is divisible by $9,$ where $D$ and $E$ each represent a single digit, define $new_variable1$ as the sum of all possible values of the sum $D+E$.
$new_variable2$ is equal to $new_variable1$.
Onum Lake has $new_variable2$ more trout than Boast Pool.   There are 75 fish in Boast Pool.  Riddle Pond has half as many fish as Onum Lake.  What is the average number of fish in all three bodies of water?

#### Output:
<The_Symbol_of_New_Variable1>
$S$
</The_Symbol_of_New_Variable1>

<The_Symbol_of_New_Variable2>
$T$
</The_Symbol_of_New_Variable2>

---

### Task:
#### Problem:
{modify_p1}
{relation}
{modify_p2}

#### Output:

"""


def main(datas: list) -> list:
    new_datas = []
    not_run = 0
    run_error = 0
    not_output = 0
    not_boxed = 0
    not_equal = 0

    for data in datas:
        # Question,solution,Answer,subject,level,unique_id
        for idx, resp in enumerate(data['responses']):
            id = data['__id__']
            if idx > 0:
                id = f'{id}_{idx}'
            
            is_run, (stdout, stderr) = data['code_state'][get_hash(resp)]
            if not is_run:
                not_run += 1
                continue
            if stderr:
                run_error += 1
                continue
            if not stdout:
                not_output += 1
                continue
            if not extract_boxed_content(stdout):
                not_boxed += 1
                continue
            
            is_queal = True if extract_boxed_content(stdout)[-1].strip().lower() == 'true' else False
            if not is_queal:
                not_equal += 1
                continue

            prompt = PROMPT.format(modify_p1=data['modify_p1'], relation=data['relationship'], modify_p2=data['modify_p2'])

            d = {
                '__id__': id,
                'prompt': f"<|user|>\n{prompt}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt}],
                'responses': [],
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
                # 'relationship_in_solution': data['relationship_in_solution'],

            }
            new_datas.append(d)
    
    print(f'not_run: {not_run}, run_error: {run_error}, not_output: {not_output}, not_boxed: {not_boxed}, not_equal: {not_equal}')
    return new_datas


if __name__ == '__main__':
    path = './project/stable_synthesize_question/result/with_code/math4500/step6.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])