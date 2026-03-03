"""
Main entry point for the multi-step inference client.

This module loads configuration from a YAML file and orchestrates
the execution of multiple inference steps for compositional problem generation.
"""

import argparse
import os
from types import SimpleNamespace

import yaml

from src.infer_multi_step_client import InferMultiStepClient
from src.logging_config import logger_manager
from src.utils import (
    check_and_create_dir,
    get_function_from_file,
    get_hash,
    get_un_generated_data,
    is_process_running,
    list_files,
    load_data,
    save_data,
    validate_function_signature,
)


def get_args() -> list[SimpleNamespace]:
    """Parse command line arguments and configuration file.

    Returns:
        List of SimpleNamespace objects, each containing configuration for a step.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='yaml config file path')
    args = parser.parse_args()

    # Load configuration from yaml file
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        common_args = SimpleNamespace(**config['common'])
        args_list = [SimpleNamespace(**args) for args in config.values()][1:]

    # Validate paths and create directories
    assert common_args.dataset_path != common_args.output_folder, "dataset_path and output_path should not be the same"
    if os.path.exists(common_args.output_folder):
        logger.warning(f"output folder {common_args.output_folder} already exists, will continue infer")
    check_and_create_dir(common_args.output_folder, is_file=False)

    # Configure each step
    for idx, args in enumerate(args_list):
        args.step_idx = idx
        args.debug = common_args.debug
        args.duplicate_num = args.duplicate_num if args.duplicate_num > 0 else 1
        # Set uniq_key from common config
        args.uniq_key = getattr(common_args, 'uniq_key', '__id__')

        # Configure generation settings based on mode
        if args.mode == 'reward':
            args.generate_config["logprobs"] = True
            args.generate_config["top_logprobs"] = 1
        elif args.mode == 'mcts':
            args.mcts_config = SimpleNamespace(**args.mcts_config)
            args.mcts_config.resp_server_names = args.resp_server_names
            args.mcts_config.debug = args.debug

        # Auto-set process number if not specified
        if args.process_num == -1:
            args.process_num = os.cpu_count()
            logger.warning(f"auto set process_num as cpu count: {args.process_num}")

        # Set input/output paths for each step
        if idx == 0:
            args.dataset_path = common_args.dataset_path
            assert os.path.exists(args.dataset_path), f"dataset_path {args.dataset_path} not exists"
        else:
            # Use previous step's temp output as current step's input
            args.dataset_path = args_list[idx-1].temp_output_folder

        args.output_folder = common_args.output_folder
        args.output_path = os.path.join(common_args.output_folder, f'step{idx}.jsonl')
        args.temp_output_folder = os.path.join(common_args.output_folder, f'step{idx}_temp')

        check_and_create_dir(args.temp_output_folder, is_file=False)
        logger.debug(f"\n     >> input {args.dataset_path}\n     >> output {args.output_path}\n     >> temp {args.temp_output_folder}")
        logger.debug(f">>> step {idx} args: {args}")

    return args_list


def main() -> None:
    """Main entry point for the inference client."""
    global logger

    # Initialize logger
    logger = logger_manager(0)

    # Parse configuration
    args_list = get_args()

    # Run multi-step inference
    infer_client = InferMultiStepClient(args_list)
    infer_client.run()

if __name__ == '__main__':
    main()