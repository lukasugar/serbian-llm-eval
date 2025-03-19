import argparse
import json
import logging
import os
from dotenv import load_dotenv
from datetime import datetime


import wandb

from lm_eval import tasks, evaluator, utils

# Load environment variables from .env file
load_dotenv()

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="Serbian")

    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--log_to_wandb", action="store_true", default=False,
                     help="Log results to Weights & Biases")

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        assert not args.provide_description  # not implemented

        if args.limit:
            print(
                "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )

        if args.tasks is None:
            task_names = tasks.ALL_TASKS
        else:
            task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

        print(f"Selected Tasks: {task_names}")

        description_dict = {}
        if args.description_dict_path:
            with open(args.description_dict_path, "r") as f:
                description_dict = json.load(f)

        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
            language=args.language,
        )

        dumped = json.dumps(results, indent=2)
        print(dumped)

        if args.output_path:
            dirname = os.path.dirname(args.output_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.output_path, "w") as f:
                f.write(dumped)

        # Log results to wandb if requested
        if args.log_to_wandb:
            try:
                log_results_to_wandb(results, args)
            except Exception as e:
                print(f"Error logging results to Weights & Biases: {e}")

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))
    except Exception as e:
        print(f"Error: {e}")
        with open("failed_runs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}\n")


def log_results_to_wandb(results, args):
    """
    Log evaluation results to Weights & Biases.
    
    For each task:
    1. Log individual prompt/response data to a "{task_name}" table 
    2. Log aggregated statistics to a "{task_name}_stats" table
    
    Args:
        results: The evaluation results dictionary
        args: Command line arguments
    """
    model_args = args.model_args.replace("pretrained=", "")
    language = args.language
    
    # Check if WANDB_API_KEY is set in environment
    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("WANDB_API_KEY not found. Please set it in your .env file.")
    
    # Get optional wandb project from environment or use default
    wandb_project = os.environ.get("WANDB_PROJECT", "serbian-llm-eval")
    
    # Initialize wandb if it's not already initialized
    if wandb.run is None:
        wandb.init(project=wandb_project, name=f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}-{model_args}-{language}-{args.tasks}")
    
    # First, check if there are write_out_info files to log individual examples
    if args.write_out and args.output_base_path:
        for task_name in results["results"].keys():
            write_out_path = os.path.join(args.output_base_path, f"{task_name}_write_out_info.json")
            if os.path.exists(write_out_path):
                with open(write_out_path, "r") as f:
                    write_out_data = json.load(f)
                
                # Create or get the task table
                table_name = task_name
                task_table = wandb.Table(columns=["model_name", "language", "json_output"])
                
                # Add each example to the table
                for item in write_out_data:
                    task_table.add_data(model_args, language, json.dumps(item))
                
                # Log the table
                wandb.log({table_name: task_table})
    
    # Log aggregated statistics for each task
    for task_name, task_results in results["results"].items():
        # Create stats table
        stats_table_name = f"{task_name}_stats"
        
        # Determine columns - model_args plus all metrics
        columns = ["model_args"] + list(task_results.keys()) + ["full_json"]
        stats_table = wandb.Table(columns=columns)
        
        # Extract row data
        row_data = [model_args]
        for metric in task_results.keys():
            row_data.append(task_results[metric])
        
        # Add the full JSON as the last column
        row_data.append(json.dumps({
            "task": task_name,
            "results": task_results,
            "config": results["config"],
            "version": results["versions"].get(task_name, "")
        }))
        
        # Add the row to the table
        stats_table.add_data(*row_data)
        
        # Log the table
        wandb.log({stats_table_name: stats_table})
    
    # Close the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
