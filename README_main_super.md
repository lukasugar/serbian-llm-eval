# main-super.py - Batch Evaluation Runner

This script allows you to run multiple language model evaluations sequentially using configuration files in YAML or JSON format. It supports multi-value parameters to make it easy to run evaluations across different languages, tasks, and models.

## Usage

```bash
python main_super.py --config path/to/config.yaml
```

or

```bash
python main_super.py --config path/to/config.json
```

## Configuration File Format

The configuration file should be in YAML or JSON format and must contain a `runs` key with a list of configurations. Each configuration in the list will be executed sequentially.

### Multi-Value Parameters

The script supports multiple values for the following parameters:
- `language`
- `model_args`
- `tasks`

When you specify a list of values for these parameters, the script will automatically generate separate runs for each combination of values (Cartesian product).

### Example YAML Configuration with Multi-Value Parameters

```yaml
runs:
  # Run with multiple languages
  - model: "hf-causal-experimental"
    model_args: "pretrained=Qwen/Qwen2.5-0.5B-Instruct"
    language: ["Serbian-Cyrillic", "Serbian", "English"]  # Will create 3 separate runs
    tasks: "openbookqa"
    device: "mps"
    num_fewshot: 0
    batch_size: 4
    output_base_path: "./results"  # Will be auto-adjusted
    write_out: true
  
  # Run with multiple tasks
  - model: "hf-causal-experimental"
    model_args: "pretrained=Qwen/Qwen2.5-0.5B-Instruct"
    language: "Serbian"
    tasks: ["arc_challenge", "arc_easy", "boolq"]  # Will create 3 separate runs
    device: "mps"
    num_fewshot: 0
    batch_size: 4
    output_base_path: "./results"
    write_out: true
    
  # Cartesian product example (2 models × 2 languages × 2 tasks = 8 separate runs)
  - model: "hf-causal-experimental"
    model_args: [
      "pretrained=Qwen/Qwen2.5-0.5B-Instruct", 
      "pretrained=meta-llama/Llama-2-7b-hf"
    ]
    language: ["Serbian-Cyrillic", "Serbian"]
    tasks: ["openbookqa", "arc_challenge"]
    device: "mps"
    num_fewshot: 0
    batch_size: 4
    output_base_path: "./results"
    write_out: true
```

### Example JSON Configuration with Multi-Value Parameters

```json
{
  "runs": [
    {
      "model": "hf-causal-experimental",
      "model_args": "pretrained=Qwen/Qwen2.5-0.5B-Instruct",
      "language": ["Serbian-Cyrillic", "Serbian", "English"],
      "tasks": "openbookqa",
      "device": "mps",
      "num_fewshot": 0,
      "batch_size": 4,
      "output_path": "./results/openbookqa.json",
      "output_base_path": "./results",
      "write_out": true
    },
    {
      "model": "hf-causal-experimental",
      "model_args": [
        "pretrained=Qwen/Qwen2.5-0.5B-Instruct",
        "pretrained=meta-llama/Llama-2-7b-hf"
      ],
      "language": ["Serbian-Cyrillic", "Serbian"],
      "tasks": ["openbookqa", "arc_challenge"],
      "device": "mps",
      "num_fewshot": 0,
      "batch_size": 4,
      "output_path": "./results/results.json",
      "output_base_path": "./results",
      "write_out": true
    }
  ]
}
```

## Output Path Adjustment

When using multi-value parameters, the script will automatically adjust output paths:

- The script creates a directory structure as follows:
  `output_base_path/<model_short_name>/<language>/<task>.json`

- Model short names are created by replacing the "/": For example, `pretrained=Qwen/Qwen2.5-0.5B-Instruct` becomes `pretrained=Qwen__Qwen2.5-0.5B-Instruct`

This automatic path adjustment ensures that each combination of model, language, and task gets its own unique output file.

## Configuration Parameters

Each run configuration supports the following parameters:

### Required Parameters

- `model`: The model to use (required)

### Optional Parameters

- `language`: Language for evaluation (default: "Serbian") - can be a string or list of strings
- `model_args`: Additional model arguments (default: "") - can be a string or list of strings
- `tasks`: Comma-separated list of tasks to evaluate - can be a string or list of strings
- `num_fewshot`: Number of few-shot examples (default: 0)
- `batch_size`: Batch size for evaluation
- `max_batch_size`: Maximum batch size to try with auto batch size
- `device`: Device to run the model on (e.g., "cuda:0", "mps", "cpu")
- `output_path`: Path to save the results JSON
- `limit`: Limit the number of examples per task
- `data_sampling`: Data sampling rate
- `description_dict_path`: Path to description dictionary
- `output_base_path`: Base path for output files
- `provide_description`: Whether to provide task descriptions (boolean flag)
- `no_cache`: Whether to disable caching (boolean flag)
- `check_integrity`: Whether to check data integrity (boolean flag)
- `write_out`: Whether to write out predictions (boolean flag)

## Error Handling

If a configuration run fails, the script will ask whether you want to continue with the next configuration or stop execution.