import sys
sys.path.append('./')
from src.utils import run_python, get_hash, is_code_safe



def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    try:
        res = []
        for i, stop_reason in enumerate(stop_reasons):
            if '```python' in resps[i]:

                code = resps[i].split('```python')[-1]
                if '```' not in code: continue
                code = code.split('```')[0].strip()

                res.append(resps[i])

                if 'code_state' not in data:
                    data['code_state'] = {}
           
                if is_code_safe(code):
                    stdout, stderr = run_python(code, timeout=5, max_retries=2)
                    data['code_state'][get_hash(resps[i])] = (True, (stdout, stderr))
                else:
                    data['code_state'][get_hash(resps[i])] = (False, ('dangerous code', 'dangerous code'))

        return res
    
    except Exception as e:
        print('filter func error:', e)
        raise e


if __name__ == '__main__':

    responses = ['```python\n# Step 1: Define the variables\nnew_variable1 = 40\nnew_variable2 = 6\n\n# Step 2: Calculate the difference\ndifference = new_variable2 - new_variable1\n\n# Step 3: Print intermediate variables and final answer\nprint("new_variable1 =", new_variable1)\nprint("new_variable2 =", new_variable2)\nprint("difference =", difference)\nprint("\\boxed{", difference, "}")\n```']
    data = {
        'responses': responses
    }

    finish_reasons = ['stop']
    stop_reasons = ['```output']

    print("Testing filter_and_execute_python function...")
    print(f"Input responses: {len(responses)}")

    # Run the main function
    result = main(data, responses, finish_reasons, stop_reasons)

    print(f"Filtered responses: {len(result)}")
    if result:
        print("Response content:", result[0][:100] + "..." if len(result[0]) > 100 else result[0])

    if 'code_state' in data:
        print(f"Code execution results: {len(data['code_state'])} entries")
        for hash_key, (success, (stdout, stderr)) in data['code_state'].items():
            print(f"  - Success: {success}")
            if stdout:
                print(f"    stdout: {stdout}")
            if stderr:
                print(f"    stderr: {stderr}")


    print("Test completed successfully!")

