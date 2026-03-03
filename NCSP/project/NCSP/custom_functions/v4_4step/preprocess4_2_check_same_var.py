import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import get_hash, load_data


def main(datas: list) -> list:
    new_datas = []
    PROMPT = """Given Math Problem 1 and Math Problem 2, confirm whether Problem 2 uses the same variable symbols or object names as Problem 1, focusing on avoiding potential confusion if the two problems were combined.

1. If the same variable symbols (e.g. x, $alpha$, etc) are present in both problems, regardless of whether they have different roles, output 'yes'.
2. If the same objects or entities  (e.g. Xiaoming's speed, the number of cakes, Triangle $ABC$, etc) are mentioned and relevant to the problems, output 'yes'.
3. Otherwise, output 'no'.

# Output Format:
<IsContain>  
{yes or no}  
</IsContain>  
<Analysis>  
{A brief analysis}  
</Analysis>

---

# Example 1:
## Problem 1: 
Xiaoming's speed is 80 km/h. Xiaohong's speed is 40 km/h. How much faster is Xiaoming than Xiaohong?
## Problem 2: 
Xiaoming can run 10 km in 1 hour. How long does it take Xiaoming to run 20 km?
## Output:  
<IsContain>  
yes
</IsContain>  
<Analysis>  
1. In Problem 1: Xiaoming's speed, Xiaohong's speed, and the speed difference are mentioned.
2. In Problem 2: Xiaoming's speed and the distance are mentioned.
Both problems mention Xiaoming's speed, which will cause confusion if the two problems are combined.
</Analysis>

---

# Example 2:
## Problem 1: 
What is the area of a rectangle $ABCD$ with length 5 and width 3?
## Problem 2: 
If a rectangle $ABCD$'s width is 3 and circumference is 16, what is the length?
## Output:
<IsContain>
yes  
</IsContain>
<Analysis>
1. In Problem 1: Rectangle $ABCD$ and its length and width are mentioned.
2. In Problem 2: Rectangle $ABCD$ and its width and circumference are mentioned.
Both problems mention rectangle $ABCD$, which will make its properties ambiguous if the two problems are combined.
</Analysis>

---

# Example 3:
## Problem 1:
Solve for z in the equation z + 5 = 10.
## Problem 2:
What is the solution to x + 5 = 10?
## Output:
<IsContain>
no
</IsContain>
<Analysis>  
1. In Problem 1: The variable symbol z is mentioned.
2. In Problem 2: The variable symbol x is mentioned.
No variable symbols are the same in both problems.
</Analysis>

---

# Example 4:
## Problem 1:
x=5, b=1, what is x+b?
## Problem 2:
Find the solution to 2x+c=6 if c=1.
## Output:
<IsContain>
yes
</IsContain>
<Analysis>
1. In Problem 1: The variable symbols x and b are mentioned.
2. In Problem 2: The variable symbols x and c are mentioned.
Both problems mention the variable symbol x, so the answer is 'yes', regardless of the different roles of x in the two problems.
</Analysis>

---

# Task
## Problem 1:
{QUESTION1}
## Problem 2:
{QUESTION2}
## Output:
"""
    # path = '/code/infer_client/data/compose_generalization/MATH4500_3FullPairSet_idx1.jsonl'
    # ids = set(data['__id__'] for data in load_data(path))

    for data in datas:
        # if data['__id__'] not in ids: continue
        id = data['__id__']
        question1 = data['data1']['problem'] + '\n' + data['definition_of_var1']
        question2 = data['modify_p2']
        resp = data['responses'][0]
        symbol_of_var1 = resp.split('<The_Symbol_of_New_Variable1>')[1].split('</The_Symbol_of_New_Variable1>')[0].strip().strip('$')
        symbol_of_var2 = resp.split('<The_Symbol_of_New_Variable2>')[1].split('</The_Symbol_of_New_Variable2>')[0].strip().strip('$')

        question1 = question1.replace('new_variable1', symbol_of_var1)
        question2 = question2.replace('new_variable2', symbol_of_var2)
        
        prompt = PROMPT.replace('{QUESTION1}', question1).replace('{QUESTION2}', question2)
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
                
                'symbol_of_var1': symbol_of_var1,
                'symbol_of_var2': symbol_of_var2,
            }
        new_datas.append(d)

    return new_datas


if __name__ == '__main__':
    # path = '/code/infer_client/data/compose_generalization/MATH4500_3FullPairSet.jsonl'
    path = '/code/infer_client/data/compose_generalization/100UniquePairsAllSuccess.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    for data in new_datas:
        print(data['messages'][0]['content'], '\n\n', '-'*100)
    # for key, value in new_datas[0].items():
    #     print(key, value, '\n')