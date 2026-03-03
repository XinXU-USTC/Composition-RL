# cd infer_client
# get the input for NCSP
# output file is ./data/MATH_full_permutations_200*199.jsonl
python3 python3 project/NCSP/custom_functions/v4_4step/pre_and_post/pre_make_seed_pair.py

# start analysis and merge problem pairs
python3 main.py --config_path project/NCSP/config/v4_4step/stable_with_code_math4500_demo.yaml

# final merge the "items" by template
python3 project/NCSP/custom_functions/v4_4step/pre_and_post/make_final_dataset.py --path project/NCSP/result/v4_4step/math4500/step10.jsonl --save_path project/NCSP/result/v4_4step/math4500/final_dataset.jsonl
