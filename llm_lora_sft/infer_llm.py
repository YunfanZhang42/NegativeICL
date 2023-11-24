import argparse
import random
import time
import pandas as pd
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
from peft import PeftModel


def fill_template(template, input):
    filled_templates = []
    for i in input:
        filled_template = template.format(input=i, target="")
        filled_templates.append(filled_template)
    return filled_templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer on a dataset using a trained LLM + LoRA model")
    parser.add_argument("--model-type", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--load-lora", type=str, default="./experiments/saved_models/mistral-7b-sft")

    parser.add_argument("--val-dataset-path", type=str, default="./path/to/val_data.parquet")
    parser.add_argument("--saved-dataset-path", type=str, default="../path/to/val_data_with_results.parquet")
    parser.add_argument("--prompt-template-path", type=str, default="./path/to/llm_sft_few_shot_prompt.txt")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200)

    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    parser.add_argument("--random-seed", type=int, default=34967803)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--flash-attn", action="store_true", help="Use Flash Attention 2.")
    parser.add_argument("--continue-from-breakpoint", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
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
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Finished initializing tokenizer")

    prompt_template = open(args.prompt_template_path, "r").read().strip()
    df = pd.read_parquet(args.val_dataset_path, engine="pyarrow")
    if "results" not in df.columns:
        df["results"] = ""
    print(f"Finished loading dataset from {args.val_dataset_path}, with {len(df)} rows.")

    # Figure out data type
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32

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

    if args.load_lora:
        lora_model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=False)
        model = lora_model.merge_and_unload()
        print(f"Finished loading Lora model from {args.load_lora}")

    model = model.eval()

    model = torch.compile(model, disable=not args.compile)
    print(f"Finished loading model and lora.")

    last_batch_time = time.time()

    for i in range(0, len(df), args.batch_size):
        batch = df["input"].iloc[i : i + args.batch_size]
        batch = batch.tolist()

        results_batch = df["results"].iloc[i : i + args.batch_size]
        results_batch = results_batch.tolist()
        if results_batch[0] != "" and args.continue_from_breakpoint:
            print(f"Skipping index {i} - {i + args.batch_size} because they have already been processed.")
            continue
        
        input_strings = fill_template(prompt_template, batch)
        inputs = tokenizer(input_strings, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)

        outputs = model.generate(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_length=args.max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        # Decode the generated ids, only keep the new text
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated_texts = [str(text).strip() for text in generated_texts]

        df.loc[i : i + args.batch_size - 1, "results"] = generated_texts

        for j, generated_text in enumerate(generated_texts):
            print(f"Index: {i + j}")
            print(f"Input text: {batch[j]}")
            print(f"Generated text: {generated_text}")
        
        print(f"Samples per second: {args.batch_size / (time.time() - last_batch_time)}")
        last_batch_time = time.time()

        if i > 0 and (i // args.batch_size) % args.save_every == 0:
            df.to_parquet(args.saved_dataset_path, engine="pyarrow")
            print(f"Saved dataset to {args.saved_dataset_path}")

    df.to_parquet(args.saved_dataset_path, engine="pyarrow")
    print(f"Saved dataset to {args.saved_dataset_path}")
