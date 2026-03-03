import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data



# Prompt 3: Difference Calculation
PROMPT_STEP3 = """**Task Description:**

Write a Python program to calculate the value of `$new_variable2$` based on the given value of `$new_variable1$` and the specified relationship. Follow these guidelines for your program:

1. Avoid using floating-point numbers for intermediate steps; instead, use the `sympy` library to handle fractions, square roots, and other symbolic representations.
2. Clearly print intermediate steps where appropriate.
3. Ensure that the output clearly shows the value of `$new_variable2$` in its most simplified form.
4. The output should include both the Python program and the corresponding output produced by running the program.
5. Print the \\boxed{final answer} at the end.

**Output Format:**

```python
{The Python code that computes `$new_variable2$`, including print statements for intermediate calculations and the final result.}
```

```output
{The output of the Python program, showing intermediate steps and the final value of \\boxed{`$new_variable2$`}.}
```

---

**Example1:**

## The value of $new_variable1$:
3

## The relationship:
$new_variable2$ is 5 more than $new_variable1$.

# Output:

```python
from sympy import symbols

# Define variables
new_variable1 = 3
diff = 5

# Calculate new_variable2, which is new_variable1 + diff
new_variable2 = new_variable1 + diff

# Output the process and result
print(f"As new_variable1 is {new_variable1} and new_variable2 is {diff} more than new_variable1, we have:")
print(f"new_variable2 = new_variable1 + diff = {new_variable1} + {diff} = {new_variable2}")
print("The final answer is \\boxed{", new_variable2, "}")
```

```output
As new_variable1 is 3 and new_variable2 is 5 more than new_variable1, we have:
new_variable2 = new_variable1 + diff = 3 + 5 = 8
The final answer is \\boxed{ 8 }
```

---

**Example2:**

## The value of $new_variable1$:
\\frac{3}{4}


## The relationship:
$new_variable2$ is 3*sqrt(2)/4 less than $new_variable1$.


# Output:

```python
from sympy import symbols, Rational, sqrt

# Define variables
new_variable1 = Rational(3, 4)
diff = Rational(3, 4) * sqrt(2)

# Calculate new_variable2, which is new_variable1 - diff
new_variable2 = new_variable1 - diff

# Output the process and result
print(f"As new_variable1 is {new_variable1} and new_variable2 is {diff} less than new_variable1, we have:")
print(f"new_variable2 = new_variable1 - diff = {new_variable1} - {diff} = {new_variable2}")
print("The final answer is \\boxed{", new_variable2, "}")
```

```output
As new_variable1 is 3/4 and new_variable2 is 3 more than new_variable1, we have:
new_variable2 = new_variable1 - diff = 3/4 - 3*sqrt(2)/4 = 3/4 - 3*sqrt(2)/4
The final answer is \\boxed{ 3/4 - 3*sqrt(2)/4 }
```

---

**Task:** 

## The value of $new_variable1$:
{NEW_VARIABLE1}


## The relationship:
{RELATIONSHIP}

"""


def main(datas: list) -> list:
    new_datas = []
    not_run = 0
    run_error = 0
    not_output = 0
    not_boxed = 0
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
                print(data['messages'][0]['content'].split("Task:")[-1])
                print(resp)
                print(stderr)
                continue
            if not stdout:
                not_output += 1
                continue

            difference = extract_boxed_content(stdout)[-1]
            if not difference:
                not_boxed += 1
                continue

            relationship = f"$new_variable2$ is {difference} more than $new_variable1$."
            
            var_of_p1 = data['var1']
    
    
            prompt = PROMPT_STEP3.replace('{NEW_VARIABLE1}', var_of_p1).replace('{RELATIONSHIP}', relationship)
            
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
                
                'code_of_p1p2': resp.split('```python\n')[-1].split('```')[0],
                'output_of_p1p2': stdout,
                "difference": difference,
                'relationship': relationship,
            }
            new_datas.append(d)
        
    return new_datas


if __name__ == '__main__':
    path = 'project/stable_synthesize_question/result/built_relationship_3step/merge_p1p2/step0.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])