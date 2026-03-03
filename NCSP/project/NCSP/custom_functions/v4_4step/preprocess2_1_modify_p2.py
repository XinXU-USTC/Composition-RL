import sys
sys.path.append('./')

from copy import deepcopy

from src.utils import extract_boxed_content, get_hash, load_data





# Prompt 2: Identify $new_variable2$ from Problem2

PROMPT_STEP2 = """Given a math problem, please identify a specific integer, float, or fraction within it and replace it with $new_variable2$. The specific steps are as follows:

Step1: Identify and list all the numerical values in the problem. There are several situations:
    1. Containing unknown variables is not allowed, for example, in the case of $2a+3 = 5$, '2', '3', and '5' is valid, while $2a+3$ isn't.
    2. Containing math symbol is allowed, do not simplify or round them to decimals, for example, keep the $3\pi$ as $3\pi$, keep the $\sqrt(2)$ as $\sqrt(2)$, keep the $7!$ as $7!$, and keep the $\frac{6}{8}$ as $\frac{6}{8}$;
    3. Containing unit is not allowed, for example, $120^\circ$ or $120°$ should be replaced by ${new_variables2}^\circ$ where ${new_variables2}=120$; $10%$ can become ${new_variable2}%$ where ${new_variables2}=10$; and 1000_{2} should choose the number of base to be defined as $1000_{new_variable2}$, etc.;
    4. Choose part of expression like coefficie, numerator or denominatornt is allowed, for example, the expression $7x + 3$ can be replaced by ${new_variable2} * x + 3$;

Step2: There are many types of the numerical values, and choose one following the priority: 
    integers like '1' > fractions like '\frac{1}{2}' or '1/2' > decimals like '0.5' > numbers in words like 'one', 'two', three' > numbers in other bases like '1000_{2}'
    Small numbers are preferred when there are multiple numerical values of the same type.
    If there are no any numerical values in the problem, output 'None' in the tag <The_Value_of_New_Variable2>.

## Output Format:

<Identify_New_Variable2>
{Identified a specific integer, float, or fraction as $new_variable2$}
</Identify_New_Variable2>

<The_Value_of_New_Variable2>
{The value of $new_variable2$, no other text, and output 'None' if there are no any numerical values in the problem}
</The_Value_of_New_Variable2>

<The_Definition_of_New_Variable2>
{The definition of $new_variable2$ without mentioning the real value, and output 'None' if there are no any numerical values in the problem}
</The_Definition_of_New_Variable2>

<Modified_Problem>
{The modified problem with the new variable symbol $new_variable2$ without mentioning the real value of $new_variable2$, and output 'None' if there are no any numerical values in the problem}
</Modified_Problem>

---

Example1:

## Input:

A box contains 6 oranges. If you add 8 more oranges, how many oranges will be in the box? (Add the original 6 oranges)

## Output:

<Identify_New_Variable2>
Step1: There are two numerical value in the problem: integer '6' and integer '8'.
Step2: Both numbers are integers, so we choose the small one, '6', as $new_variable2$, so $new_variable2$ = 6.
</Identify_New_Variable2>

<The_Value_of_New_Variable2>
    6
</The_Value_of_New_Variable2>

<The_Definition_of_New_Variable2>
    Define $new_variable2$ as the initial number of oranges in the box.
</The_Definition_of_New_Variable2>

<Modified_Problem>
    A box contains $new_variable2$ oranges. If you add 8 more oranges, how many oranges will be in the box? (Add the original $new_variable2$ oranges)
</Modified_Problem>

---

Example2:

## Input:

Let $a,$ $b,$ $c$ be positive real numbers such that
\[\log_a b + \log_b c + \log_c a = 0.\]Find
\[(\log_a b)^3 + (\log_b c)^3 + (\log_c a)^3.\]

## Output:

<Identify_New_Variable2>
Step1: There are many numerical value in the problem: 0 (the right side of the equation \[\log_a b + \log_b c + \log_c a = 0.\]), 3 (the power of the expression \[(\log_a b)^3 + (\log_b c)^3 + (\log_c a)^3.\]), and even 1 (the hidden coefficient of the expression of unknowns).
Step2: Both numbers are integers and has the same priority, but the number 0 is the smallest and more simple, so we choose the number 0 as $new_variable2$, so $new_variable2$ = 0.
</Identify_New_Variable2>

<The_Value_of_New_Variable2>
    0
</The_Value_of_New_Variable2>

<The_Definition_of_New_Variable2>
    Define the result of \[\log_a b + \log_b c + \log_c a\] as $new_variable2$.
</The_Definition_of_New_Variable2>

<Modified_Problem>
    Let $a,$ $b,$ $c$ be positive real numbers such that
\[\log_a b + \log_b c + \log_c a = {new_variable2}.\]Find
\[(\log_a b)^3 + (\log_b c)^3 + (\log_c a)^3.\]
</Modified_Problem>

---

Example3:

## Input:

Which area is bigger, a circle or a square of the same circumference?

## Output:

<Identify_New_Variable2>
Step1: There is no explicit numerical value in this question. So we need to try to dig out some hidden numerical values. 
    We found that the question asked "a circle or a square", which can be converted to "1 circle or 1 square".
    Therefore, the numerical values contained in the question are: 1 (the number of circles), 1 (the number of squares).
Step2: Both numbers are '1', so we choose the number of circles, '1', as $new_variable2$, so $new_variable2$ = 1.
</Identify_New_Variable2>

<The_Value_of_New_Variable2>
    1
</The_Value_of_New_Variable2>

<The_Definition_of_New_Variable2>
    Define $new_variable2$ as the number of circles in the question.
</The_Definition_of_New_Variable2>

<Modified_Problem>
    Which area is bigger, $new_variable2$ circles or a square of the same circumference?
</Modified_Problem>

---

Task:

## Input:

{QUESTION}

## Output:
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
            data1 = data['data1']
            data2 = data['data2']
            question2 = data2['problem']

            resp = data['responses'][0]

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

            prompt_step2 = PROMPT_STEP2.replace('{QUESTION}', question2)
            d2 = {
                '__id__': data['__id__'],
                'prompt': f"<|user|>\n{prompt_step2}<|assistant|>\n",
                'messages': [{'role': 'user', 'content': prompt_step2}],
                'responses': [],
                'data1': data1,
                'data2': data2,
                'modify_p1': data['modify_p1'],
                'var1': data['var1'],
                'definition_of_var1': data['definition_of_var1'],
                'analysis1': data['analysis1'],
            }
            new_datas.append(d2)
    
    print(f'not_run: {not_run}, run_error: {run_error}, not_output: {not_output}, not_boxed: {not_boxed}, not_equal: {not_equal}')
    return new_datas


if __name__ == '__main__':
    path = '/code/infer_client/project/stable_synthesize_question/result/with_code/math4500/step2.jsonl'
    datas = load_data(path)
    new_datas = main(datas)
    print(len(new_datas))
    print(new_datas[0])