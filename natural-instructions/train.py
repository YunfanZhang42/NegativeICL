import argparse
import time
import random
import json
import os

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
from data import NaturalInstructionsV1Seq2SeqDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./dev_config.json", help="Path to config file")
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
    train_dataset = NaturalInstructionsV1Seq2SeqDataset(
        subtask_dir=config.subtask_dir,
        tasks=config.train_tasks,
        template=config.template,
        pos_template=config.pos_template,
        neg_template=config.neg_template,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length,
        positive_examples=config.positive_examples,
        negative_examples=config.negative_examples,
        additional_instructions=config.additional_instructions,
    )

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
    )
    print(f"Finished loading datasets, training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

    # Initialize the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // config.gradient_accumulation_steps,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
    valid_dataloader = DataLoader(
        val_dataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=True, num_workers=config.num_workers
    )

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

    # if the load_model path doesn't exist, then we assume we're starting from scratch
    if not os.path.exists(config.load_model):
        torch.save(model.state_dict(), config.load_model)
    if config.load_model is not None:
        model.load_state_dict(torch.load(config.load_model, map_location=device))
    if config.activation_checkpointing:
        model.gradient_checkpointing_enable()
    print(f"Finished loading model")

    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config.cycle_steps,
        cycle_mult=1.0,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        gamma=config.lr_gamma,
    )
    print("Initialized optimizer.")

    best_eval_loss = 10.0**9

    model.train()
    scaler = GradScaler(enabled=config.fp16)

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            with autocast(device_type=str(device), dtype=torch.float16, enabled=config.fp16):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss /= config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_count += 1
            batch_loss += float(loss.item())

            if step_count % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                batch_count += 1
                print(
                    f"Epoch {epoch}, Batch {batch_count}, Loss {batch_loss}, samples/sec {config.batch_size / (time.time() - last_batch_time)}"
                )
                log_writer.add_scalar("train_loss", batch_loss, batch_count)
                batch_loss = 0.0
                last_batch_time = time.time()

                if batch_count > 0 and batch_count % config.eval_very_n_steps == 0:
                    model.eval()
                    eval_loss = 0
                    eval_start_time = time.time()
                    with torch.no_grad():
                        for batch in valid_dataloader:
                            with autocast(device_type=str(device), dtype=torch.float16, enabled=config.fp16):
                                inputs = batch["input_ids"].to(device)
                                labels = batch["labels"].to(device)
                                mask = batch["attention_mask"].to(device)

                                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                            eval_loss += float(outputs.loss.item())

                    eval_loss /= len(valid_dataloader)
                    log_writer.add_scalar("eval_loss", eval_loss, batch_count)
                    print(
                        f"Batch count {batch_count}, Eval loss {eval_loss}, "
                        + f"samples/sec {len(valid_dataloader) * config.eval_batch_size / (time.time() - eval_start_time)}"
                    )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        torch.save(model.state_dict(), f"./checkpoints/{config.model_name}_best.pt")

                    torch.save(model.state_dict(), f"./checkpoints/{config.model_name}_latest.pt")
                    print(f"Saving latest model with loss {eval_loss}, batch {batch_count}")

                    model.train()

    print(f"Finished training for {config.num_epochs} epochs, {batch_count} batches with best eval loss {best_eval_loss}")
