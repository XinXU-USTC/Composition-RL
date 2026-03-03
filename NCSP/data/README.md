# Data Directory

## Overview
This directory contains seed datasets used as the input of the NCSP (Numerical-based Composition Synthesis Pipeline).


## MATH Dataset Seeds

### Download MATH Dataset
``` bash
# data/MATH/test-00000-of-00001.parquet
# data/MATH/train-00000-of-00001.parquet
https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark/tree/main/data
```

### Get The Input Pairs of NCSP
```bash
# Generate seed data from MATH dataset
python3 project/NCSP/custom_functions/v4_4step/pre_and_post/pre_make_seed_pair.py
```

### Output Files
- **`MATH_seeds_200.jsonl`**: 200 unique math problems extracted from MATH dataset
  - 40 problems per difficulty level (Level 1-5)
  - Each entry contains: `__id__`, `level`, `type`, `problem`, `solution`, `final_answer`, `solution_answer`
  - `__id__` is generated using `get_hash(problem)` from `src/utils.py`

- **`MATH_full_permutations_200*199.jsonl`**: All permutations of seed pairs (39,800 pairs)
  - Used as input for the NCSP framework
  - Each entry contains paired data with `data1` and `data2`

### Usage
Use `MATH_full_permutations_200*199.jsonl` as the input for the NCSP framework to generate compositional math problems.


## VCMD Dataset

### File: `VCMD.jsonl`
Contains 500 records of processed compositional math problems with human annotations and system-generated intermediate results.

VCMD is sampled from ECMD and annotated by human annotators.

### Data Structure

Each record contains the following fields in JSON format:

```json
{
  // ====================================
  // First-level fields: Human-verified/final results
  // ====================================
  "__id__": "Unique identifier for the record (hash of content)",
  "idx": "Sequential index (0-499)",
  "question1": "Original first problem from seed data",
  "question2": "Original second problem from seed data",
  "merged_question": "Final merged/composed problem",
  "final_problem": "The final processed problem (same as merged_question)",
  "var1": "Value of variable 1 (extracted and verified)",
  "var2": "Value of variable 2 (extracted and verified)",
  "symbol_of_var1": "Symbol representation of variable 1 (e.g., 'N')",
  "symbol_of_var2": "Symbol representation of variable 2 (e.g., 'M')",
  "Answer": "Final answer to the composed problem",

  // Human verification flags
  "human_is_p1_complete": "Human verification if problem 1 processing is complete",
  "human_is_p2_complete": "Human verification if problem 2 processing is complete",
  "human_check_var1": "Human verification for variable 1 extraction",
  "human_check_var2": "Human verification for variable 2 extraction",
  "human_check_same_var": "Human verification for variable consistency",
  "human_repaired": "Human corrections (TRUE/FALSE or actual modify content)",

  // ====================================
  // Base data: Complete processing pipeline (system-generated)
  // ====================================
  "base_data": {
    "__id__": "Combined ID of the seed pair",
    "messages": "Messages used for API calls",

    // Input seed data
    "data1": {
      "__id__": "Seed ID for first problem",
      "level": "Difficulty level (Level 1-5)",
      "type": "Math subject type",
      "problem": "Problem statement",
      "solution": "Solution explanation",
      "final_answer": "Final numerical answer",
      "solution_answer": "Answer in solution text"
    },
    "data2": {
      "__id__": "Seed ID for second problem",
      "level": "Difficulty level (Level 1-5)",
      "type": "Math subject type",
      "problem": "Problem statement",
      "solution": "Solution explanation",
      "final_answer": "Final numerical answer",
      "solution_answer": "Answer in solution text"
    },

    // Step 1-2: Problem modification
    "modify_p1": "Modified version of problem 1 with variable placeholders",
    "modify_p2": "Modified version of problem 2 with variable placeholders",
    "var1": "Extracted variable 1 value (raw extraction)",
    "var2": "Extracted variable 2 value (raw extraction)",
    "definition_of_var1": "Definition of variable 1 in the context",
    "definition_of_var2": "Definition of variable 2 in the context",
    "analysis1": "Analysis of variable 1 extraction",
    "analysis2": "Analysis of variable 2 extraction",

    // Step 3: Difference calculation
    "code_of_p1p2": "Python code to calculate difference between variables",
    "output_of_p1p2": "Output of the difference calculation",
    "relationship": "Textual relationship between variables",

    // Step 4: Solution generation
    "code_of_s1s2": "Python code for generating solutions",
    "output_of_s1s2": "Output of solution generation",
    "var2_by_calculation": "Variable 2 value calculated from solutions",
    "symbol_of_var1": "Symbol assigned to variable 1",
    "symbol_of_var2": "Symbol assigned to variable 2"
  },
}
```

