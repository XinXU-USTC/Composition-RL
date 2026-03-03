import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data

# Prompt 3: Difference Calculation
PROMPT_STEP3 = """Given the values of $new_variable1$ and $new_variable2$, your task is to calculate the difference $new_variable2 - new_variable1$.

Write a Python program that calculates the difference between $new_variable2$ and $new_variable1$. The program should follow these guidelines:
    1. Instead of writing functions, write programs directly.
    2. Avoid using decimal values, and ensure that all fractions and square roots are simplified using functions from the `sympy` library.
    3. If there are any intermediate variables, print them in the Python programs.
    4. Print the \\boxed{final answer} at the end.
    5. Provide the code output following the Python code.

## Output Format:

```python
{The Python code that computes `$new_variable2 - new_variable1$`, including print statements for intermediate variables and the final answer.}
```

```output
{The output of the Python code, including intermediate variables and the final answer.}
```

---

**Example 1:**

# The values of $new_variable1$:
\\frac{3}{4}

# The values of $new_variable2$:
$6

# Output:

```python
from sympy import Rational

# Step 1: Define the variables
new_variable1 = Rational(3, 4)
new_variable2 = Rational(6)

# Step 2: Calculate the difference
difference = new_variable2 - new_variable1

# Step 3: Print intermediate variables and final answer
print("new_variable1 =", new_variable1)
print("new_variable2 =", new_variable2)
print("difference =", difference)
print("\\boxed{", difference, "}")
```

```output
new_variable1 = 3/4
new_variable2 = 6
difference = 21/4
\\boxed{ 21/4 }
```

---

**Example 2:**

# The values of $new_variable1$:
\sqrt{2}

# The values of $new_variable2$:
7!

# Output:

```python
from sympy import sqrt, factorial

# Step 1: Define the variables
new_variable1 = sqrt(2)
new_variable2 = factorial(7)

# Step 2: Calculate the difference
difference = new_variable2 - new_variable1

# Step 3: Print intermediate variables and final answer
print("new_variable1 =", new_variable1)
print("new_variable2 =", new_variable2)
print("difference =", difference)
print("\\boxed{", difference, "}")
```

```output
new_variable1 = sqrt(2)
new_variable2 = 5040
difference = 5040 - sqrt(2)
\\boxed{ 5040 - sqrt(2) }
```

---

Task:

# The values of $new_variable1$:
{NEW_VARIABLE1}

# The values of $new_variable2$:
{NEW_VARIABLE2}

"""


def main(datas: list) -> list:
    new_datas = []
    p1_or_p2_is_none = 0
    not_run = 0
    run_error = 0
    not_output = 0
    not_boxed = 0
    not_equal = 0
    print(f"len datas {datas}")
    for data in datas:
        # Question,solution,Answer,subject,level,unique_id
        resp = data['responses'][0]
        id = data['__id__']

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
            # print(data['data1']['final_answer'], '\n', data['messages'][0]['content'].split('Task:**')[-1],'\n\n',data['modify_p1'])
            continue

        modify_p2 = data['modify_p2']
        var2 = data['var2']
        analysis2 = data['analysis2']
        # definition_of_var2 = resp.split('<The_Definition_of_New_Variable2>')[1].split('</The_Definition_of_New_Variable2>')[0].strip()
        definition_of_var2 = data['definition_of_var2']

        var1 = data['var1']
        modify_p1 = data['modify_p1']
        if any([var.lower() == 'none' for var in [var1, var2, modify_p1, modify_p2]]):
            p1_or_p2_is_none += 1
            continue
            
        prompt = PROMPT_STEP3.replace('{NEW_VARIABLE1}', var1).replace('{NEW_VARIABLE2}', var2)
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

            'modify_p2': modify_p2,
            'var2': var2,
            'definition_of_var2': definition_of_var2,
            'analysis2': analysis2,
        }
        new_datas.append(d)
        
    print(f'p1_or_p2_is_none: {p1_or_p2_is_none}')
    return new_datas


if __name__ == '__main__':
    path = 'project/stable_synthesize_question/result/built_relationship_3step/modify_p1/step0.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])