import argparse
import time
import random
import os
import json

from dotmap import DotMap
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from data import NaturalInstructionsV1Seq2SeqDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./config_llm_test.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    # Set up the environment
    # log_writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, args.model_name))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up device: {device}")

    # Initialize the tokenizer
    if "llama" in args.model_type:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer")

    # Initialize the datasets
    train_dataset = NaturalInstructionsV1Seq2SeqDataset(
        subtask_dir=args.subtask_dir,
        tasks=args.train_tasks,
        template=args.template,
        pos_template=args.pos_template,
        neg_template=args.neg_template,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        positive_examples=args.positive_examples,
        negative_examples=args.negative_examples,
        additional_instructions=args.additional_instructions,
        lm_type="decoder_only",
        max_context_length=args.max_context_length,
    )

    val_dataset = NaturalInstructionsV1Seq2SeqDataset(
        subtask_dir=args.subtask_dir,
        tasks=args.val_tasks,
        template=args.template,
        pos_template=args.pos_template,
        neg_template=args.neg_template,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        positive_examples=args.positive_examples,
        negative_examples=args.negative_examples,
        additional_instructions=args.additional_instructions,
        lm_type="decoder_only",
        max_context_length=args.max_context_length,
    )

    # Initialize the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps,
        shuffle=True,
        drop_last=True,
        # Training LLMs will be slow anyway, so we don't need to use multiple workers
        num_workers=1,
    )
    valid_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True, num_workers=1)
    print(f"Finished loading datasets, training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

    # Initialize the model and optimizer
    # Figure out data type
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    if "llama" in args.model_type:
        model = LlamaForCausalLM.from_pretrained(
            args.model_type, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn
        )
    elif "mistral" in args.model_type or "zephyr" in args.model_type:
        model = MistralForCausalLM.from_pretrained(
            args.model_type, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_type, torch_dtype=torch_dtype)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.load_lora:
        target_modules = args.lora_modules.split(",")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )

        lora_model = get_peft_model(model, lora_config)
        print(f"Finished initializing Lora model")
    else:
        lora_model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=True)
        print(f"Finished loading Lora model from {args.load_lora}")

    lora_model.print_trainable_parameters()
    model = torch.compile(lora_model, disable=not args.compile)

    print(f"Finished loading model and lora")

    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, fused=True)
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

    best_eval_loss = 10.0**9

    model.train()
    scaler = GradScaler(enabled=(args.dtype == "float16"))

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(args.num_epochs):
        for batch in train_dataloader:
            # Note that we don't need AMP for bf16 and fp32
            with autocast(device_type=str(device), dtype=torch.float16, enabled=(args.dtype == "float16")):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss /= args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_count += 1
            batch_loss += float(loss.item())

            if step_count % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                batch_count += 1
                print(
                    f"Epoch {epoch}, Batch {batch_count}, Loss {batch_loss}, samples/sec {args.batch_size / (time.time() - last_batch_time)}"
                )
                # log_writer.add_scalar("train_loss", batch_loss, batch_count)
                batch_loss = 0.0
                last_batch_time = time.time()

                if batch_count > 0 and batch_count % args.eval_very_n_steps == 0:
                    model.eval()

                    model.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_latest"))
                    print(f"Saved latest model, batch {batch_count}")

                    eval_loss = 0
                    eval_start_time = time.time()
                    with torch.no_grad():
                        for batch in valid_dataloader:
                            with autocast(device_type=str(device), dtype=torch.float16, enabled=(args.dtype == "float16")):
                                inputs = batch["input_ids"].to(device)
                                labels = batch["labels"].to(device)
                                mask = batch["attention_mask"].to(device)

                                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                            eval_loss += float(outputs.loss.item())

                    eval_loss /= len(valid_dataloader)
                    # log_writer.add_scalar("eval_loss", eval_loss, batch_count)
                    print(
                        f"Batch count {batch_count}, Eval loss {eval_loss}, "
                        + f"samples/sec {len(valid_dataloader) * args.eval_batch_size / (time.time() - eval_start_time)}"
                    )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        model.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_best"))
                        print(f"Saved best model, batch {batch_count}")

                    model.train()

    # Save the final model at the end of training.
    model.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "latest"))
    print(f"Finished training for {args.num_epochs} epochs, {batch_count} batches with best eval loss {best_eval_loss}")
