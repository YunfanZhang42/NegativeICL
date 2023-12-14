import argparse
import time
import random
import json

import numpy as np
import torch
import torch.optim as optim
from dotmap import DotMap
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    LongT5ForConditionalGeneration,
    BartTokenizerFast,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from data import NaturalInstructionsV1Seq2SeqDataset, HF_LOSS_IGNORE_TOKEN_ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./generate_config.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        config = DotMap(json.load(f))

    # Set up the environment
    log_writer = SummaryWriter(f"./logs/{config.model_name}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up device: {device}")

    # Initialize the tokenizer
    if "flan-t5" in config.model_type or "google/t5" in config.model_type:
        tokenizer = T5TokenizerFast.from_pretrained(config.model_type)
    elif "bart" in config.model_type:
        tokenizer = BartTokenizerFast.from_pretrained(config.model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    print(f"Finished initializing tokenizer")

    # Initialize the datasets
    val_dataset = NaturalInstructionsV1Seq2SeqDataset(
        subtask_dir=config.subtask_dir,
        tasks=config.val_tasks,
        template=config.template,
        pos_template=config.pos_template,
        neg_template=config.neg_template,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length,
        positive_examples=config.positive_examples,
        negative_examples=config.negative_examples,
        additional_instructions=config.additional_instructions,
        mode="eval",
    )
    print(f"Finished loading datasets, evaluation samples: {len(val_dataset)}")

    # Initialize the dataloaders
    valid_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # Initialize the model and optimizer
    if "flan-t5" in config.model_type or "google/t5" in config.model_type:
        model = T5ForConditionalGeneration.from_pretrained(config.model_type).to(device)
    elif "google/long-t5" in config.model_type:
        model = LongT5ForConditionalGeneration.from_pretrained(config.model_type).to(device)
    elif "bart" in config.model_type:
        model = BartForConditionalGeneration.from_pretrained(config.model_type).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_type).to(device)

    model = torch.compile(model, disable=not config.compile)
    
    if config.load_model is not None:
        model.load_state_dict(torch.load(config.load_model, map_location=device))
    if config.activation_checkpointing:
        model.gradient_checkpointing_enable()
    print(f"Finished loading model")

    results = {}
    model.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            with autocast(device_type=str(device), dtype=torch.float16, enabled=config.fp16):
                inputs = batch["input_ids"].to(device)
                output = model.generate(
                    inputs=inputs,
                    max_new_tokens=config.max_output_length,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
                result = tokenizer.decode(output[0], skip_special_tokens=True)
                print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
                if batch["task_name"][0] not in results:
                    results[batch["task_name"][0]] = {"pred": [], "gt": []}
                results[batch["task_name"][0]]["pred"].append(result)
                results[batch["task_name"][0]]["gt"].append(batch["all_outputs"][0])
                print("Task:", batch["task_name"][0])
                print("Ground truth:")
                print("\n".join(batch["all_outputs"][0]))
                print("Predicted:")
                print(result)
                print("-" * 64)
    # Save the results to disk
    with open(f"./generation_results/{config.model_name}.json", "w") as f:
        json.dump(results, f, indent=2)
