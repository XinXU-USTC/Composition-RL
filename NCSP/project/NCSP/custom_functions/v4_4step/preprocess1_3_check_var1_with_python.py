import random
import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data


PROMPT_CHECK = """**Task Description:**

Write a Python program to compare two given values and determine if they are equal. Follow these guidelines:

1. Use the `sympy` library to handle symbolic comparisons, ensuring that equivalent expressions (e.g., \( \frac{2}{4} \) and \( \frac{1}{2} \)) are recognized as equal.
2. For values involving irrational constants (e.g., \( \pi \), \( e \)), perform comparisons up to **two decimal places** for practical equivalence.
3. Include clear intermediate steps in the program, such as evaluating or simplifying the values where appropriate.
4. Wrap the final comparison outcome in a `\\boxed{}` command for clarity.
5. Provide both the Python code and the results of running the code.

**Output Format:**

```python
{The Python code that compares the two given values, including print statements for intermediate steps and the \\boxed{final comparison outcome}.}
```

```output
{The output of the Python program.}
```

---

**Example 1:**

### Value1: 15
### Value2: 17_8

```python
from sympy import Eq

# Define the values
value1 = 15  # Decimal representation
value2 = int('17', 8)  # Octal to decimal conversion

# Print intermediate steps
print(f"Value1: {value1}")
print(f"Value2: {value2}")

# Check if the values are equal
are_equal = Eq(value1, value2)
print(f"Are the values equal? \\boxed{{{are_equal}}}")
```

```output
Value1: 15
Value2: 15
Are the values equal? \\boxed{True}
```

---

**Example 2:**

### Value1: \( 5! \)
### Value2: \( \sqrt{120} \times \sqrt{24} \)


```python
from sympy import Eq, factorial, sqrt

# Define the values
value1 = factorial(5)
value2 = sqrt(120) * sqrt(24)

# Simplify Value2 for clarity
simplified_value2 = value2.simplify()

# Print intermediate steps
print(f"Value1: {value1}")
print(f"Simplified Value2: {simplified_value2}")

# Check if the values are equal
are_equal = Eq(value1, simplified_value2)
print(f"Are the values equal? \\boxed{{{are_equal}}}")
```

```output
Value1: 120
Simplified Value2: 24*sqrt(5)
Are the values equal? \\boxed{False}
```

---

**Example 3:**

### Value1: \( 2\pi \)
### Value2: \( 6.28 \)

```python
from sympy import Eq, pi

# Define the values
value1 = 2 * pi
value2 = 6.28

# Print intermediate steps
print(f"Value1: {value1}")
print(f"Value2: {value2}")

# Check if the values are equal (to two decimal places)
are_equal = abs(value1 - value2) < 0.01
print(f"Are the values equal? \\boxed{{{are_equal}}}")
```

```output
Value1: 2*pi
Value2: 6.28
Are the values equal? \\boxed{True}
```

---

**Task:**  

### Value1: {VALUE1}  
### Value2: {VALUE2}  

"""


def main(datas: list) -> list:
    new_datas = []
    for data in datas:
        for idx, resp in enumerate(data['responses']):
            id = data['__id__']
            if idx > 0:
                id = f"{id}_{idx}"

            var1 = data['var1']
            var1_of_extracted = extract_boxed_content(resp)
            if not var1_of_extracted:
                continue

            prompt = PROMPT_CHECK.replace('{VALUE1}', var1).replace('{VALUE2}', var1_of_extracted[-1])
            d1 = {
                '__id__': id,
                # 'prompt': f"<|user|>\n{prompt}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt}],
                'responses': [],
                'data1': data['data1'],
                'data2': data['data2'],
                'modify_p1': data['modify_p1'],
                'var1': data['var1'],
                'definition_of_var1': data['definition_of_var1'],
                'analysis1': data['analysis1'],
            }

            var1 = var1.strip().strip('$').strip()
            var1_of_extracted = var1_of_extracted[-1].strip().strip('$').strip()
            if var1.lower() == var1_of_extracted.lower() or \
            var1.replace(",", "") == var1_of_extracted or var1 == var1_of_extracted.replace(",", ""):
                d1['responses'] = ["""```python\nprint(f"Are the values equal? \\boxed{{{True}}}")```\n\n"""]
                d1['code_state'] = {get_hash(d1['responses'][0]): (True, ('Are the values equal? \\boxed{True}', ''))}

            new_datas.append(d1)
        
    return new_datas


if __name__ == '__main__':
    path = '/code/infer_client/project/stable_synthesize_question/result/with_code/math4500/step1.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])