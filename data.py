import argparse
import time
import json
import os
import random

import numpy as np
from collections import Counter
from dotmap import DotMap
from torch.utils.data import Dataset
from transformers import T5TokenizerFast, BartTokenizerFast, AutoTokenizer, LlamaTokenizer


HF_LOSS_IGNORE_TOKEN_ID = -100


class NaturalInstructionsV1Seq2SeqDataset(Dataset):
    def __init__(
        self,
        subtask_dir="./natural-instructions-1.0/data/",
        tasks=[],
        template="./templates/natural_instructions_v1_template.txt",
        pos_template="./templates/natural_instructions_v1_positive_examples_template.txt",
        neg_template="./templates/natural_instructions_v1_negative_examples_template.txt",
        tokenizer=None,
        padding="max_length",
        max_input_length=1024,
        max_output_length=128,
        additional_instructions=True,
        positive_examples=2,
        negative_examples=2,
        mode="train",
        lm_type="encoder_decoder",
        max_context_length=2048,
    ):
        super().__init__()

        self.sample_inputs = []
        self.sample_outputs = []
        # TODO: use a random choice from all outputs during training.
        self.sample_all_outputs = []
        self.sample_task_names = []
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.padding = padding
        self.max_input_length = max_input_length
        self.tokenizer.model_max_length = max_input_length
        self.max_output_length = max_output_length
        self.additional_instructions = additional_instructions
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.mode = mode
        self.lm_type = lm_type
        self.max_context_length = max_context_length

        if self.lm_type =="decoder_only":
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset statistics:
        # Key: task name
        # Value: another dictionary with the following keys:
        # num_positive_examples: a list of number of positive examples actually used in the task
        # num_negative_examples: a list of number of negative examples actually used in the task
        # num_input_context_len_exceeded: number of times the context length was exceeded, even without any examples
        # num_output_context_len_exceeded: number of times the output context length was exceeded, even without any examples
        self.dataset_statistics = {}

        for task in tasks:
            task_path = os.path.join(subtask_dir, task)
            with open(task_path, "r") as f:
                task_data = json.load(f)
            self.add_task_to_samples(task, task_data, template, pos_template, neg_template)

        # For each of the tasks, print out the dataset statistics as follows:
        # Task name
        # Total number of examples
        # Number of positive examples used, by all distinct number of positive examples
        # Number of negative examples used, by all distinct number of negative examples
        # Number of times the context length was exceeded, even without any examples
        # Number of times the output context length was exceeded, even without any examples
        for task_name, task_statistics in self.dataset_statistics.items():
            print(f"Task name: {task_name}")
            print(f"Total number of examples: {len(task_statistics['num_positive_examples'])}")

            positive_example_counts = Counter(task_statistics["num_positive_examples"])
            negative_example_counts = Counter(task_statistics["num_negative_examples"])

            print("Number of positive examples used, by all distinct number of positive examples:")
            for num, count in positive_example_counts.items():
                print(f"    {num}: {count}")

            print("Number of negative examples used, by all distinct number of negative examples:")
            for num, count in negative_example_counts.items():
                print(f"    {num}: {count}")

            print(
                f"Number of times the context length was exceeded, even without any examples: {task_statistics['num_input_context_len_exceeded']}"
            )
            print(
                f"Number of times the output context length was exceeded, even without any examples: {task_statistics['num_output_context_len_exceeded']}"
            )
            print()

    def format_examples(self, template, example_data):
        # Shuffle the examples, format them to strings, while also calculating their lengths.
        random.shuffle(example_data)
        examples_str_and_lengths = []
        for i, example in enumerate(example_data):
            reason = example["reason"] if "reason" in example else "N/A"
            suggestion = example["suggestion"] if "suggestion" in example else "N/A"
            example_str = template.format(
                example_index=str(i + 1), input=example["input"], output=example["output"], reason=reason, suggestion=suggestion
            )
            example_length = len(self.tokenizer(example_str, padding=False)["input_ids"])
            examples_str_and_lengths.append((example_str, example_length))
        return examples_str_and_lengths

    def add_task_to_samples(self, task_name, task_data, template, pos_template, neg_template):
        task_name = task_name.split(".")[0]

        self.dataset_statistics[task_name] = {
            "num_positive_examples": [],
            "num_negative_examples": [],
            "num_input_context_len_exceeded": 0,
            "num_output_context_len_exceeded": 0,
        }

        # Read the templates
        with open(template, "r") as f:
            template = f.read().strip()
        with open(pos_template, "r") as f:
            pos_template = f.read().strip()
        with open(neg_template, "r") as f:
            neg_template = f.read().strip()

        for sample in task_data["Instances"]:
            # First, calculate the length of the input without any examples
            input_without_examples = template.format(
                title=task_data["Title"],
                definition=task_data["Definition"],
                prompt=task_data["Prompt"],
                avoid=task_data["Things to Avoid"],
                emphasis=task_data["Emphasis & Caution"],
                negative_examples="",
                positive_examples="",
                input=sample["input"],
            )

            # Calculate the length of the input without any examples
            input_without_examples_length = len(self.tokenizer(input_without_examples, padding=False)["input_ids"])
            # Check if the input length is exceeded without any examples
            if input_without_examples_length > self.max_input_length:
                self.dataset_statistics[task_name]["num_input_context_len_exceeded"] += 1

            # Next, shuffle and format the examples
            positive_examples = []
            if (
                "Positive Examples" in task_data["Examples"]
                and task_data["Examples"]["Positive Examples"]
                and task_data["Examples"]["Positive Examples"][0] != "-"
            ):
                positive_examples = self.format_examples(pos_template, task_data["Examples"]["Positive Examples"])
                positive_examples.reverse()

            negative_examples = []
            if (
                "Negative Examples" in task_data["Examples"]
                and task_data["Examples"]["Negative Examples"]
                and task_data["Examples"]["Negative Examples"][0] != "-"
            ):
                negative_examples = self.format_examples(neg_template, task_data["Examples"]["Negative Examples"])
                negative_examples.reverse()

            # While the length is not exceeded, add alternating positive and negative examples
            selected_positive_examples = []
            selected_negative_examples = []
            current_input_length = input_without_examples_length

            while (len(positive_examples) > 0 and len(selected_positive_examples) < self.positive_examples) or (
                len(negative_examples) > 0 and len(selected_negative_examples) < self.negative_examples
            ):
                if len(positive_examples) > 0 and len(selected_positive_examples) < self.positive_examples:
                    next_positive_example = positive_examples.pop()
                    next_positive_example_str = next_positive_example[0]
                    next_positive_example_length = next_positive_example[1]
                    if current_input_length + next_positive_example_length > self.max_input_length:
                        break
                    else:
                        current_input_length += next_positive_example_length
                        selected_positive_examples.append(next_positive_example_str)
                if len(negative_examples) > 0 and len(selected_negative_examples) < self.negative_examples:
                    next_negative_example = negative_examples.pop()
                    next_negative_example_str = next_negative_example[0]
                    next_negative_example_length = next_negative_example[1]
                    if current_input_length + next_negative_example_length > self.max_input_length:
                        break
                    else:
                        current_input_length += next_negative_example_length
                        selected_negative_examples.append(next_negative_example_str)

            # Add the number of positive and negative examples actually used to the dataset statistics
            self.dataset_statistics[task_name]["num_positive_examples"].append(len(selected_positive_examples))
            self.dataset_statistics[task_name]["num_negative_examples"].append(len(selected_negative_examples))

            # Format the final input with the selected examples.
            positive_examples_str = "\n".join(selected_positive_examples)
            negative_examples_str = "\n".join(selected_negative_examples)

            input_with_examples = template.format(
                title=task_data["Title"].strip(),
                definition=task_data["Definition"].strip(),
                prompt=task_data["Prompt"].strip(),
                avoid=task_data["Things to Avoid"].strip(),
                emphasis=task_data["Emphasis & Caution"].strip(),
                negative_examples=negative_examples_str.strip(),
                positive_examples=positive_examples_str.strip(),
                input=sample["input"],
            )

            # Randomly select a reference output.
            output = random.choice(sample["output"])

            outputs_tokenized = self.tokenizer(
                output, padding=self.padding, max_length=self.max_output_length, truncation=True, return_overflowing_tokens=True
            )
            if len(outputs_tokenized["overflow_to_sample_mapping"]) > 1:
                self.dataset_statistics[task_name]["num_output_context_len_exceeded"] += 1

            self.sample_inputs.append(input_with_examples)
            self.sample_outputs.append(output)
            self.sample_all_outputs.append(sample["output"])
            self.sample_task_names.append(task_name)

    def __len__(self):
        return len(self.sample_inputs)

    def __getitem__(self, idx):
        if self.lm_type == "encoder_decoder":
            inputs = self.tokenizer(
                self.sample_inputs[idx], padding=self.padding, max_length=self.max_input_length, truncation=True, return_tensors="np"
            )
            outputs = self.tokenizer(
                self.sample_outputs[idx], padding=self.padding, max_length=self.max_output_length, truncation=True, return_tensors="np"
            )
            labels = outputs["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = HF_LOSS_IGNORE_TOKEN_ID

            return {
                "input_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten(),
                # Hugging Face's Seq2Seq models needs a labels argument, and should be able to generate decoder attention mask automatically.
                "labels": labels.flatten(),
                "all_outputs": self.sample_all_outputs[idx] if self.mode != "train" else False,
                "task_name": self.sample_task_names[idx],
            }
        elif self.lm_type == "decoder_only":
            text_encoded = self.tokenizer(
                self.sample_inputs[idx],
                truncation=True,
                max_length=self.max_input_length,
                padding=False,
                return_tensors="np",
                add_special_tokens=False,
            )

            text_target_encoded = self.tokenizer(
                self.sample_outputs[idx],
                truncation=True,
                max_length=self.max_output_length,
                padding=False,
                return_tensors="np",
                add_special_tokens=False,
            )

            # Add start token to the beginning of the text_encoded, with attention mask of 1
            text_input_ids = np.insert(text_encoded["input_ids"], 0, self.tokenizer.bos_token_id)
            text_attention_mask = np.insert(text_encoded["attention_mask"], 0, 1)

            # Add end token to the end of the text_target_encoded, with attention mask of 1
            text_target_input_ids = np.append(text_target_encoded["input_ids"], self.tokenizer.eos_token_id)
            text_target_attention_mask = np.append(text_target_encoded["attention_mask"], 1)

            # Combine the two encodings
            input_ids = np.append(text_input_ids, text_target_input_ids)
            attention_mask = np.append(text_attention_mask, text_target_attention_mask)

            # Truncate the combined encoding to max_context_length by removing tokens from the left side
            input_ids = input_ids[-self.max_context_length :]
            attention_mask = attention_mask[-self.max_context_length :]

            # Pad the combined encoding to max_length, on the left side, with padding token
            input_ids = np.pad(input_ids, (self.max_context_length - len(input_ids), 0), constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (self.max_context_length - len(attention_mask), 0), constant_values=0)

            # Now, mark the last len(text_target_input_ids) tokens as labels, and the rest as -100
            labels = np.full(self.max_context_length, HF_LOSS_IGNORE_TOKEN_ID)
            labels[-len(text_target_input_ids) :] = text_target_input_ids

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input": self.sample_inputs[idx],
                "target": self.sample_outputs[idx],
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./config_llm_test.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    # Initialize the tokenizer
    if "llama" in args.model_type:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer")

    # Initialize the datasets
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

    print(f"Finished loading datasets")

    for i in range(100):
        print("Sample Index:", i)
        batch = val_dataset[i]
        input_str = tokenizer.decode(batch["input_ids"])
        output_ids = [token_id if token_id != HF_LOSS_IGNORE_TOKEN_ID else tokenizer.pad_token_id for token_id in batch["labels"]]
        print("Input:\n", tokenizer.decode(batch["input_ids"]))
        print("Output:\n", tokenizer.decode(output_ids))
        num_non_padded_tokens = len([token_id for token_id in batch["input_ids"] if token_id != tokenizer.pad_token_id])
        print("Number of non-padded tokens in inputs:", num_non_padded_tokens)
        num_non_padded_tokens = len([token_id for token_id in output_ids if token_id != tokenizer.pad_token_id])
        print("Number of non-padded tokens in outputs:", num_non_padded_tokens)
