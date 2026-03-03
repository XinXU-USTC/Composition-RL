import random
import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data


PROMPT_STEP1 = """Given a math problem and the final answer, your task is to findout one number from the answer and provide the corresponding definition. Follow the steps below:

Step1: Identify a specific integer, float, or fraction within `final_answer` and name it as $new_variable1$; There are several situations:
    
    1. If the `final_answer` contains unknown variables:
        1.1 If the `final_answer` is a expression, choose one coefficient as $new_variable1$, for example, 2x + 3, you can choose the coefficient of x as $new_variable1$, which is 2, and in the case of sin(x), there is a hidden coefficient 1 and a hidden amplitude 1, you can choose either one as $new_variable1$;
        1.2 If the `final_answer` is a equation, you can choose one solution as $new_variable1$, for example, y = 2x + 1, you can define the value of $y$ as $new_variable1$ when given x = 1, which is 3;
        1.3 If the `final_answer` is a symbol of an option or a word, such as 'A', 'B', 'CAT', etc, use their first letter's order in the alphabet as a variable, such as 'A' = 1, 'B' = 2, 'CAT' = 3, etc;
        1.4 If the `final_answer` contains 2 or more items, e.g. multiple choice questions, choose the smallest or the largest one, and then apply the corresponding situation;

    2. If the `final_answer` has no unknown variables, there are several situations:
        2.1 If the `final_answer` itself is a numerical value, like 'four', '4', '$2 + \sqrt{2}$', '$3\pi$', and '$\frac{3}{4}$', use it directly as $new_variable1$;
        2.2 If the `final_answer` contains 2 or more numerical values, use the largest or the smallest one as $new_variable1$;
        2.3 If the `final_answer` is an interval or ratio, choose one boundary and \infty is not allowed, for example, [2,\infty), you can define the lower bound as $new_variable1$, which is 2;
        2.4 If the `final_answer` is a ratio, choose one part of the ratio, for example, 3:4, you can define the first part of the simplified ratio as $new_variable1$, which is 3;
        2.5 If the `final_answer` is a non-base 10 number, for example, 1001_{2}, you can define 'the number of digits in the base 2 representation' as $new_variable1$, which is 4;
        2.6 If the `final_answer` is an angle or degree, choose the corresponding radian value, for example, 30^\cric or 30°, define the corresponding radian value of final answer as $new_variable1$, which is \pi/6 or π/6.

    All in all, find a way to identify a specific numerical value as $new_variable1$ without unknown, and make sure reader can get the value of $new_variable1$ from the `final_answer` through your definition.

Step2: Output the value of $new_variable1$, keep the exact value or math symbol, and simplify the fraction if necessary, for example, keep the $\pi$ as $\pi$, keep the $\sqrt{2}$ as $\sqrt{2}$, and simplify \frac{6}{8} as \frac{3}{4}, without rounding to a decimal point

Step3: Output the definition of $new_variable1$ without mentioning the real value.

---

Output Format:

<Analysis>
    {Identified a specific integer, float, or fraction as $new_variable1$}
</Analysis>

<The_Value_of_New_Variable1>
    {The value of $new_variable1$, no other text, and output 'None' if you can not find a suitable $new_variable1'}
</The_Value_of_New_Variable1>

<The_Definition_of_New_Variable1>
    {The definition of $new_variable1$ without mentioning the real value, and output 'None' if you can not find a suitable $new_variable1'}
</The_Definition_of_New_Variable1>

---

Example1:

# Input:

## Original Problem: 
A baker has 12 apples. Simplified the ratio of the remaining apples to the original number of apples if he uses 3 apples to make a pie.

## `final_answer` of Problem:
\\frac{3}{4}

# Output:

<Analysis>
The `final_answer` contains no unknown variables and itself is a numerical value, match the situation 2.1. 
So, we can directly use it as the value of $new_variable1$. Therefore, $new_variable1$ = \\frac{3}{4}.
</Analysis>

<The_Value_of_New_Variable1>
\\frac{3}{4}
</The_Value_of_New_Variable1>

<The_Definition_of_New_Variable1>
Define $new_variable1$ is the result of the problem.
</The_Definition_of_New_Variable1>

---

Example2:

# Input:

## Original Problem:
Solve the equation \(x^2 - 5x + 6 = b\) if \(b = 0\).

## `final_answer` of Problem:
\(x_1 = 2\), \(x_2 = 3\)

# Output:

<Analysis>
The `final_answer` contains no unknown variables, and contains two numerical values: 2 and 3, match the situation 2.2. 
So we choose the smaller number '2' as $new_variable1$. Therefore, $new_variable1$ = 2.
</Analysis>

<The_Value_of_New_Variable1>  
2  
</The_Value_of_New_Variable1>

<The_Definition_of_New_Variable1>
Define the smaller solution of the equation as $new_variable1$.
</The_Definition_of_New_Variable1>

---

Example3:

# Input:

## Original Problem:
Simplify the expression \(3(x + 2) + 4(x + 1)\).

## `final_answer` of Problem:
7x + 10

# Output:

<Analysis>
The `final_answer` contains unknown variables and it is a expression 7x + 10, match the situation 1.1. 
So, we choose the coefficient '7' as $new_variable1$. Therefore, $new_variable1$ = 7.
</Analysis>

<The_Value_of_New_Variable1>  
7
</The_Value_of_New_Variable1>

<The_Definition_of_New_Variable1>
Define $new_variable1$ is the coefficient of x in the simplified expression.
</The_Definition_of_New_Variable1>

---

Example4:

# Input:

## Original Problem:
One acute angle of a right triangle is 60 degrees. Find the other acute angle.

## `final_answer` of Problem:
30^\circ

# Output:

<Analysis>
The `final_answer` contains no unknown variables and it is the value 30^\circ in degrees, match the situation 2.6. So, we can convert it to radians first: 30^\circ = 30 * \frac{\pi}{180} = \frac{\pi}{6}. Therefore, $new_variable1$ = \frac{\pi}{6}.
</Analysis>

<The_Value_of_New_Variable1>  
\frac{\pi}{6}
</The_Value_of_New_Variable1>

<The_Definition_of_New_Variable1>
Define $new_variable1$ is the radian value corresponding to the other acute angle.
</The_Definition_of_New_Variable1>

---

Task:

## Original Problem:
{QUESTION}

## `final_answer` of Problem:
{FINAL_ANSWER}

"""



def main(datas: list) -> list:
    new_datas = []
    for data in datas:
            data1 = data['data1']
            data2 = data['data2']
            # Question,solution,Answer,subject,level,unique_id
            question1 = data1['problem']
            question2 = data2['problem']
            final_answer1 = data1['final_answer']
            final_answer2 = data2['final_answer']
            
            prompt_step1 = PROMPT_STEP1.replace('{QUESTION}', question1).replace('{FINAL_ANSWER}', final_answer1)

            d1 = {
                '__id__': data['__id__'],
                'prompt': f"<|user|>\n{prompt_step1}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt_step1}],
                'responses': [],
                'data1': data1,
                'data2': data2,
            }
            new_datas.append(d1)
        
    return new_datas


if __name__ == '__main__':
    path = '/code/infer_client/data/MATH7500_pair_gpt4o.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])