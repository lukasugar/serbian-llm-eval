import argparse
from datetime import datetime
import json
import logging
import os
import subprocess
import sys
import yaml
from typing import Dict, List, Any, Union, Optional
import itertools


logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluations based on config file")
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML or JSON config file specifying models, languages, and tasks"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configurations from YAML or JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if file_ext == '.yaml' or file_ext == '.yml':
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")
        elif file_ext == '.json':
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON config: {e}")
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}. Use .yaml, .yml, or .json")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration file structure"""
    if 'runs' not in config:
        raise ValueError("Config file must contain a 'runs' key with a list of configurations")
    
    if not isinstance(config['runs'], list):
        raise ValueError("The 'runs' key must contain a list of configurations")
    
    for i, run_config in enumerate(config['runs']):
        if 'model' not in run_config:
            raise ValueError(f"Run configuration at index {i} is missing required 'model' field")


def _hf_model_name_to_short_name(model_name: str) -> str:
    """Convert a model name to a short name"""
    model_name = model_name.replace("pretrained=", "")
    model_name = model_name.replace("/", "__")
    return model_name

def _hf_short_name_to_hf_name(short_name: str) -> str:
    """Convert a short name to a model name"""
    model_name = short_name.replace("__", "/")
    model_name = f"pretrained={model_name}"
    return model_name


def expand_multi_value_configs(run_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand configurations with multiple values for specific parameters into multiple separate configurations.
    Handles lists for 'model_args', 'language', and 'tasks'.
    """
    expanded_configs = []
    
    for config in run_configs:
        # Identify parameters that may have multiple values
        multi_params = {}
        for param in ['model_args', 'language', 'tasks']:
            if param in config and isinstance(config[param], list):
                multi_params[param] = config[param]
        
        # If no multi-value parameters, just add the original config
        if not multi_params:
            expanded_configs.append(config.copy())
            continue
            
        # Create all combinations of multi-value parameters
        param_names = list(multi_params.keys())
        param_values = [multi_params[name] for name in param_names]
        
        # Generate all combinations
        for combo in itertools.product(*param_values):
            # Create a new config with the specific combination
            new_config = config.copy()
            
            # Set the specific values for this combination
            for i, param_name in enumerate(param_names):
                new_config[param_name] = combo[i]

            # Set the output paths
            new_config['output_base_path'] = os.path.join(new_config['output_base_path'], _hf_model_name_to_short_name(new_config["model_args"]), new_config["language"])
            new_config['output_path'] = os.path.join(new_config['output_base_path'], f"{new_config['tasks']}.json")
            
            expanded_configs.append(new_config)
    
    return expanded_configs


def build_command(run_config: Dict[str, Any], main_script: str = "main.py") -> List[str]:
    """Build the command to run main.py with the specified configuration"""
    cmd = [sys.executable, main_script]
    
    # Required parameter
    cmd.extend(["--model", run_config["model"]])
    
    # Optional parameters with their respective flags
    param_mapping = {
        "language": "--language",
        "model_args": "--model_args",
        "tasks": "--tasks",
        "num_fewshot": "--num_fewshot",
        "batch_size": "--batch_size",
        "max_batch_size": "--max_batch_size",
        "device": "--device",
        "output_path": "--output_path",
        "limit": "--limit",
        "data_sampling": "--data_sampling",
        "description_dict_path": "--description_dict_path",
        "output_base_path": "--output_base_path",
    }
    
    # Boolean flags
    bool_flags = {
        "provide_description": "--provide_description",
        "no_cache": "--no_cache",
        "check_integrity": "--check_integrity",
        "write_out": "--write_out",
        "log_to_wandb": "--log_to_wandb",
    }
    
    # Add parameters to command if they exist in the config
    for param_name, flag in param_mapping.items():
        if param_name in run_config and run_config[param_name] is not None:
            cmd.extend([flag, str(run_config[param_name])])
    
    # Add boolean flags if they exist and are True
    for param_name, flag in bool_flags.items():
        if param_name in run_config and run_config[param_name]:
            cmd.append(flag)
            
    return cmd


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Validate config
    validate_config(config)
    
    # Get configurations and expand multi-value parameters
    base_runs = config['runs']
    runs = expand_multi_value_configs(base_runs)
    
    print(f"Expanded {len(base_runs)} base configurations into {len(runs)} evaluation runs")
    
    # Run each configuration sequentially
    for i, run_config in enumerate(runs):
        print(f"\n{'='*80}")
        print(f"Running configuration {i+1}/{len(runs)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if 'language' in run_config:
            print(f"Language: {run_config['language']}")
        if 'model_args' in run_config:
            print(f"Model args: {run_config['model_args']}")
        if 'tasks' in run_config:
            print(f"Tasks: {run_config['tasks']}")
        print(f"{'='*80}\n")
        
        # Build command
        cmd = build_command(run_config)
        
        # Print command for debugging
        print(f"Running command: {' '.join(cmd)}\n")
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
            print(f"\nConfiguration {i+1}/{len(runs)} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\nError running configuration {i+1}/{len(runs)}: {e}")
            # if i < len(runs) - 1:
            #     user_input = input("Continue with next configuration? (y/n): ")
            #     if user_input.lower() != 'y':
            #         break

            with open("failed_runs.txt", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration {i+1}/{len(runs)} failed: {e}\n")
                
    print("\nAll configurations completed")


if __name__ == "__main__":
    main()
