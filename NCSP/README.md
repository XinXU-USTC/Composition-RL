# Inference Client for Multi-Step Compositional Problem Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official code repository for EMNLP 2025 paper: ["From A and B to A+B: Can Large Language Models Solve Compositional Math Problems?"](https://aclanthology.org/2025.emnlp-main.660/)

## 📋 Overview

This repository implements a multi-step inference pipeline for generating compositional mathematical problems. The system uses large language models (LLMs) to create complex problems through a structured 11-step process that includes problem modification, variable extraction, solution generation, and validation.

### Key Features

- **Multi-step Pipeline**: 11-step compositional problem generation process
- **Parallel Processing**: Multi-process and coroutine-based execution for efficiency
- **Configurable Inference**: Support for various LLM models via vLLM/TGI APIs
- **Data Validation**: Built-in Python execution for mathematical validation
- **Extensible Architecture**: Custom preprocessing and filtering functions for each step

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with required dependencies
2. **LLM Server**: Deploy a Llama-3.1-70B model using vLLM
3. **Configuration**: Update model API endpoint in config file

### Installation

```bash
# Clone the repository
git clone https://github.com/Git-Shaw/infer_client.git
cd infer_client

# Install dependencies (create a requirements.txt if needed)
pip install -r requirements.txt  # Add requirements.txt with dependencies
```

### Basic Usage

1. **Deploy LLM Server**:
```bash
# Example vLLM deployment command
vllm serve meta-llama/Llama-3.1-70B-Instruct --host 0.0.0.0 --port 9090
```

2. **Configure Model Endpoint**:
Edit the configuration file at `project/NCSP/config/v4_4step/stable_with_code_math4500.yaml`:
```yaml
resp_urls:
  - "http://0.0.0.0:9090/v1"
resp_server_names:
  - "Llama-3.1-70B-Instruct"
resp_api_keys:
  - "test"
```

3. **Run Inference**:
```bash
python main.py --config_path project/NCSP/config/v4_4step/stable_with_code_math4500.yaml
```

You can also run the pipeline using a shell script:

```bash
bash main.sh
```

## 📁 Project Structure

```
infer_client/
├── main.py                          # Main entry point
├── src/
│   ├── infer_multi_step_client.py   # Multi-step orchestration
│   ├── infer_one_step_client.py     # Single step execution
│   ├── utils.py                     # Utility functions
│   ├── logging_config.py            # Logging configuration
│   └── custom_functions/            # Response processing functions
├── project/NCSP/
│   ├── config/                      # Configuration files
│   │   └── v4_4step/
│   │       └── stable_with_code_math4500.yaml
│   └── custom_functions/
│       └── v4_4step/                # Step-specific preprocessing functions
├── data/                           # Input datasets
└── project/stable_synthesize_question/result/  # Output results
```

## 🔧 Configuration

The system uses YAML configuration files to define:

- **Common Settings**: Input/output paths, debug mode
- **Step Configuration**: Processing mode, function paths, API settings
- **Generation Parameters**: Temperature, top_p, max_tokens, etc.
- **Parallel Processing**: Process numbers and coroutines per process

### Configuration Example

```yaml
common:
  debug: 0
  dataset_path: data/MATH_full_permutations_200*199.jsonl
  output_folder: project/stable_synthesize_question/result/v4_4step/math4500/

0:  # Step configuration
  mode: "chat"
  duplicate_num: 1
  preprocess_func: project/stable_synthesize_question/custom_functions/v4_4step/preprocess1_1_modify_p1.py
  resp_filter_func: project/stable_synthesize_question/custom_functions/v4_4step/filter_modify_p1.py
  resp_urls: ["http://0.0.0.0:9090/v1"]
  resp_server_names: ["Llama-3.1-70B-Instruct"]
  generate_config:
    temperature: 0.1
    top_p: 0.7
    max_tokens: 4096
```

## 🏗️ Pipeline Architecture

### 11-Step Compositional Problem Generation (Steps 0-10)

The pipeline consists of 11 individual steps, each processing data independently:

- **Step 0**: Problem 1 modification
- **Step 1-2**: Validation of problem 1

- **Step 3**: Problem 2 modification
- **Step 4-5**: Validation of problem 2

- **Step 6**: Calculating variable relationships
- **Step 7-8**: Validation of variable relationships

- **Step 9**: Variable renaming and final composition
- **Step 10**: Additional validation variable name conflict

Each step can be configured independently with its own preprocessing function, model settings, and validation criteria.

### Processing Modes

- **`chat`**: Chat completion API (vLLM)
- **`completion`**: Text completion API (TGI)
- **`wait`**: Debug mode - sleeps instead of making API calls for step-by-step testing
- **`reward`**: Reward model mode with logprobs
- **`mcts`**: Monte Carlo Tree Search mode

## 📊 Data Format

### Input Dataset Format (JSONL)
The input should contain paired data for compositional problem generation:

```json
{
  "__id__": "unique_identifier_for_pair",
  "data1": {
    "__id__": "seed_id_1",
    "level": "Level 1-5",
    "type": "Algebra/Geometry/etc.",
    "problem": "First mathematical problem statement",
    "solution": "Step-by-step solution for first problem",
    "final_answer": "Numerical answer",
    "solution_answer": "Answer in solution text"
  },
  "data2": {
    "__id__": "seed_id_2",
    "level": "Level 1-5",
    "type": "Algebra/Geometry/etc.",
    "problem": "Second mathematical problem statement",
    "solution": "Step-by-step solution for second problem",
    "final_answer": "Numerical answer",
    "solution_answer": "Answer in solution text"
  }
}
```

**Example Input File**: `data/MATH_full_permutations_200*199.jsonl`
- Contains 39,800 pairs generated from 200 unique seeds
- Each pair represents a combination for compositional problem generation

### Output Format
The final output file contains the complete compositional problem with all processing steps:

```json
{
  "__id__": "combined_id_of_seed_pair",
  "messages": [
    {
      "role": "user",
      "content": "The complete compositional problem combining both original problems with variables and their relationships"
    },
    {
      "role": "assistant",
      "content": "The complete solution to the compositional problem, including variable analysis and final answer"
    }
  ],
  "data1": {
    "__id__": "seed_id_1",
    "level": "Level 1-5",
    "type": "Algebra/Geometry/etc.",
    "problem": "First original problem statement",
    "solution": "Step-by-step solution for first problem",
    "final_answer": "Numerical answer of first problem",
    "solution_answer": "Answer in solution text of first problem"
  },
  "data2": {
    "__id__": "seed_id_2",
    "level": "Level 1-5",
    "type": "Algebra/Geometry/etc.",
    "problem": "Second original problem statement",
    "solution": "Step-by-step solution for second problem",
    "final_answer": "Numerical answer of second problem",
    "solution_answer": "Answer in solution text of second problem"
  },
  // Variable extraction and modification
  "modify_p1": "First problem with variable placeholder (new_variable1)",
  "var1": "Extracted value of variable 1",
  "definition_of_var1": "Definition of new_variable1 in context",
  "analysis1": "Analysis of how variable 1 was extracted",
  "modify_p2": "Second problem with variable placeholder (new_variable2)",
  "var2": "Extracted value of variable 2",
  "definition_of_var2": "Definition of new_variable2 in context",
  "analysis2": "Analysis of how variable 2 was extracted",
  // Variable relationship calculation
  "code_of_p1p2": "Python code to calculate relationship between variables",
  "output_of_p1p2": "Output showing variable calculation and difference",
  "relationship": "Textual description of relationship between variables (e.g., 'new_variable2 is X more than new_variable1')",
  "difference": "The calculated difference between variables",
  // Solution generation with variables
  "code_of_s1s2": "Python code to generate solution using the variables",
  "output_of_s1s2": "Generated solution incorporating variable relationships",
  "var2_by_calculation": "Variable 2 value as calculated from solutions",
  // Variable symbols
  "symbol_of_var1": "Symbol assigned to variable 1 (e.g., 'X', 'θ')",
  "symbol_of_var2": "Symbol assigned to variable 2 (e.g., 'Y', 'φ')",
  // Final answer of the compositional problem
  "final_answer": "Final numerical answer of the compositional problem"
}
```

## 🛠️ Development

### Adding Custom Steps

1. Create preprocessing function in `project/NCSP/custom_functions/v4_4step/`
2. Create corresponding filter function
3. Update configuration YAML with new step
4. Add to step sequence in configuration

### Custom Functions Interface

```python
def main(data_list):
    """
    Preprocess data for inference step.

    Args:
        data_list: List of data dictionaries

    Returns:
        List of processed data ready for LLM inference
    """
    # Your preprocessing logic here
    return processed_data
```

### Debugging

Use `debug: N` in configuration to limit processing to the first N examples for debugging. 
`debug: 0` will process all tasks. 
for example:
```yaml
common:
  debug: 10  # Limit to first 10 examples for debugging
```

## 📝 Logging

The system provides comprehensive logging with configurable levels:
- **INFO**: Progress and status updates
- **DEBUG**: Detailed execution information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

Logs include timing information, error rates, and processing statistics.

## ⚙️ Performance Tuning

### Parallel Processing
```yaml
process_num: 8           # Number of parallel processes, coroutine_num = process_num * coroutine_per_process
coroutine_per_process: 16 # Coroutines per process, coroutine_num = process_num * coroutine_per_process
max_try_per_dataset: 2    # Retry attempts for dataset items
max_try_per_request: 2    # Retry attempts for API requests
```

### API Configuration
```yaml
api_timeout: 360         # Request timeout in seconds
generate_config:
  temperature: 0.1       # Lower for more deterministic output
  top_p: 0.7           # Nucleus sampling parameter
  max_tokens: 4096      # Maximum response length
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



## 📚 Citation

If you use this code in your research, please cite our EMNLP 2025 paper:

```bibtex
@inproceedings{xiao-zhao-2025-b,
    title = "From A and {B} to {A}+{B}: Can Large Language Models Solve Compositional Math Problems?",
    author = "Xiao, Xisheng  and
      Zhao, Hanlin",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.660/",
    pages = "13068--13089",
    ISBN = "979-8-89176-332-6"
}
```

**Paper Link**: [From A and B to A+B: Can Large Language Models Solve Compositional Math Problems?](https://aclanthology.org/2025.emnlp-main.660/)

**Abstract**: Large language models (LLMs) have demonstrated strong performance in solving math problems, and there is growing research on evaluating their robustness. Unlike previous studies that create problem variants by adding perturbations to a single problem, this paper focuses on the interaction between problems. Specifically, we combine two original problems with a logical connection to get a new math problem, and measure the LLMs' performance on it to evaluate its compositional generalization, which is an important and essential reasoning capability in human intelligence. The result of experiments that cover 14 different LLMs shows that even when the mathematical essence remains unchanged, a simple form of combination can significantly reduce the performance of LLMs, revealing the limitation of their generalization ability.

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the configuration examples in `project/NCSP/config/`
- Review the custom function templates in `project/NCSP/custom_functions/`
