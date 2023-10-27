import argparse
import time
import random
import numpy as np
import torch
import deepspeed
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
# from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from data import HFCLMDataset
from deepspeed.ops.adam import DeepSpeedCPUAdam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 on WikiText-2.")
    parser.add_argument("--model-type", type=str, default="gpt2-large", help="Type of the model to train.")
    parser.add_argument("--model-name", type=str, default="gpt2-large-finetuned", help="Name of the model to train.")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load model from.")

    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of the input sequence.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Number of steps for gradient accumulation.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--max-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5 * 0.01)
    parser.add_argument("--cycle-steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)

    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--eval-batch-count", type=int, default=1_000, help="Evaluate model every X batches.")

    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training.")
    parser.add_argument("--compile", action="store_true", help="Compile the model for faster training.")
    parser.add_argument("--activation-checkpointing", action="store_true", help="Use activation checkpointing to save memory")

    parser.add_argument("--seed", type=int, default=456970, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training. (cuda, cpu)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for the dataloader.")

    args = parser.parse_args()

    # Set up the environment
    # log_writer = SummaryWriter(f"./logs/{args.model_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up device: {device}")

    # Set up ZeRO-Offload.
    DEEPSPEED_CONFIG = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {
            "enabled": args.fp16,
            "auto_cast": args.fp16,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
    }


    # 1. Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer")

    # Initialize the datasets
    train_dataset = HFCLMDataset("wikitext", "wikitext-2-raw-v1", tokenizer, dataset_type="train", max_length=args.max_length)
    valid_dataset = HFCLMDataset("wikitext", "wikitext-2-raw-v1", tokenizer, dataset_type="validation", max_length=args.max_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)

    print(f"Finished loading datasets")

    # Initialize the model and optimizer
    model = GPT2LMHeadModel.from_pretrained(args.model_type)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    model = torch.compile(model, disable=not args.compile)
    model.train()
    print(f"Finished loading model")

    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.max_lr, adamw_mode=True, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.cycle_steps,
        cycle_mult=1.0,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        gamma=args.lr_gamma,
    )
    print("Initialized optimizer.")

    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        lr_scheduler=scheduler,
        config=DEEPSPEED_CONFIG,
        # Have to enable this for ZeRO-Offload, even if we only have one GPU.
        dist_init_required=True,
    )
    print(f"Finished initializing DeepSpeed")

    best_eval_loss = 10.0**9

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(args.num_epochs):
        for batch_count, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model_engine(input_ids=inputs, attention_mask=mask, labels=labels)

            loss = outputs.loss
            
            model_engine.backward(loss)

            model_engine.step()

            if (batch_count + 1) % args.gradient_accumulation_steps == 0 and (batch_count // args.gradient_accumulation_steps) % args.eval_batch_count == 0:
                print(f"Batch {batch_count} | Loss: {loss.item()} | samples/sec: {args.batch_size / (time.time() - last_batch_time)}")
                last_batch_time = time.time()

    print(f"Finished training for {args.num_epochs} epochs, {batch_count} batches with best eval loss {best_eval_loss}")
