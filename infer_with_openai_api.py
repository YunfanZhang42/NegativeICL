import argparse
import time
import json

from dotmap import DotMap
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import BartTokenizerFast, T5TokenizerFast, AutoTokenizer

from data import NaturalInstructionsV1Seq2SeqDataset


OPENAI_API_KEY = "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"

openai_model = "gpt-3.5-turbo-1106"
max_tokens = 128
temperature = 1e-6
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0

checkpoint_interval = 1000


def openai_api_call(client, openai_system_prompt, openai_few_shot_prompt):
    messages = [{"role": "system", "content": openai_system_prompt}, {"role": "user", "content": openai_few_shot_prompt}]

    while True:
        try:
            # Make OpenAI API call.
            response = client.chat.completions.create(
                model=openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            # Extract assistant response.
            assistant_response = response.choices[0].message.content.strip()

            # Debug
            print(f"Input: \n {openai_few_shot_prompt}")
            print(f"Assistant response: {assistant_response}")
            return assistant_response
        except Exception as e:
            print(f"Exception: {e}, retrying...")
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        config = DotMap(json.load(f))

    # Load the system prompt
    with open(config.openai_system_prompt, "r") as f:
        openai_system_prompt = f.read().strip()

    # Initialize the tokenizer
    if "flan-t5" in config.model_type or "google/t5" in config.model_type:
        tokenizer = T5TokenizerFast.from_pretrained(config.model_type)
    elif "bart" in config.model_type:
        tokenizer = BartTokenizerFast.from_pretrained(config.model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    print(f"Finished initializing tokenizer")

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

    # Initialize the OpenAI API
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

    results = {}
    for i, batch in enumerate(valid_dataloader):
        input_str = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        result = openai_api_call(client, openai_system_prompt, input_str)
        if batch["task_name"][0] not in results:
            results[batch["task_name"][0]] = {"pred": [], "gt": []}
        results[batch["task_name"][0]]["pred"].append(result)
        results[batch["task_name"][0]]["gt"].append(batch["all_outputs"][0])

        if i % checkpoint_interval == 0:
            # Save the results
            with open(f"./generation_results/{config.model_name}.json", "w") as f:
                json.dump(results, f, indent=2)
