import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data


# Prompt 2: Identify $new_variable2$ from Problem2

PROMPT = """Given two math problems, Problem 1 and Problem 2, where a numerical value in Problem 1 has been replaced by a variable $new_variable2$ to form Problem 2, your task is to identify the value of $new_variable2$.

Output the results with the following format:

<The_Value_of_New_Variable2>
{The value of $new_variable2$, if identified, or 'None' if no value can be determined.}
</The_Value_of_New_Variable2>

---

# Example 1:

## Input:

### Original Problem 1:
A box contains 10 apples. If you add 5 more apples, how many apples will there be in the box?

### Problem 2:
A box contains $new_variable2$ apples. If you add 5 more apples, how many apples will there be in the box?

## Output:

<The_Value_of_New_Variable2>  
10  
</The_Value_of_New_Variable2>

---

# Example 2:

## Input:

### Original Problem 1:
Solve for \(x\): \(3x + 4 = 16\)

### Problem 2:
Solve for \(x\): \(3x + $new_variable2$ = 16\)

## Output:

<The_Value_of_New_Variable2>  
4  
</The_Value_of_New_Variable2>

---

# Example 3:

## Input:

### Original Problem 1:
The length of a rectangle is 12 cm, and the width is 8 cm. What is the area of the rectangle?

### Problem 2:
The length of a rectangle is $new_variable1$ cm, and the width is 8 cm. What is the area of the rectangle?

## Output:

<The_Value_of_New_Variable2>
None
</The_Value_of_New_Variable2>

---

# Task:

## Input:

### Original Problem 1:
{QUESTION1}

### Problem 2:
{QUESTION2}

## Output:

"""


def main(datas: list) -> list:
    new_datas = []

    for data in datas:
        # Question,solution,Answer,subject,level,unique_id
            data1,data2 = data['data1'],data['data2']

            origin_question = data2['problem']
            resp = data['responses'][0]
            var2 = resp.split('<The_Value_of_New_Variable2>')[1].split('</The_Value_of_New_Variable2>')[0].strip()
            if not var2 or var2 == 'None':
                continue
            var2_definition = resp.split('<The_Definition_of_New_Variable2>')[1].split('</The_Definition_of_New_Variable2>')[0].strip()
            modify_p2 = resp.split('<Modified_Problem>')[1].split('</Modified_Problem>')[0].strip()
            analysis2 = resp.split('<Identify_New_Variable2>')[1].split('</Identify_New_Variable2>')[0].strip()

            prompt = PROMPT.replace('{QUESTION1}', origin_question).replace('{QUESTION2}', modify_p2)

            d2 = {
                '__id__': data['__id__'],
                'prompt': f"<|user|>\n{prompt}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt}],
                'responses': [],
                'data1': data1,
                'data2': data2,
                'modify_p1': data['modify_p1'],
                'var1': data['var1'],
                'definition_of_var1': data['definition_of_var1'],
                'analysis1': data['analysis1'],

                'modify_p2': modify_p2,
                'var2': var2,
                'definition_of_var2': var2_definition,
                'analysis2': analysis2,
            }
            new_datas.append(d2)
    
    return new_datas


if __name__ == '__main__':
    path = '/code/infer_client/data/compose_generalization/100UniquePairsAllSuccess.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0]['prompt'])