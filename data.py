import argparse
import time
import json
import os
import random
from dotmap import DotMap
from torch.utils.data import Dataset
from transformers import T5TokenizerFast, BartTokenizerFast


HF_LOSS_IGNORE_TOKEN_ID = -100


class NaturalInstructionsV1Seq2SeqDataset(Dataset):
    def __init__(
        self,
        subtask_dir="./natural-instructions-1.0/data/",
        tasks=[],
        template="./natural_instructions_v1_template.txt",
        pos_template="./natural_instructions_v1_positive_examples_template.txt",
        neg_template="./natural_instructions_v1_negative_examples_template.txt",
        tokenizer=None,
        padding="max_length",
        max_input_length=1024,
        max_output_length=128,
    ):
        super().__init__()

        self.sample_inputs = []
        self.sample_outputs = []
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.padding = padding
        self.max_input_length = max_input_length
        self.tokenizer.model_max_length = max_input_length
        self.max_output_length = max_output_length

        for task in tasks:
            task_path = os.path.join(subtask_dir, task)
            with open(task_path, "r") as f:
                task_data = json.load(f)
            self.add_task_to_samples(task_data, template, pos_template, neg_template)

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

    def add_task_to_samples(self, task_data, template, pos_template, neg_template):
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

            input_without_examples_length = len(self.tokenizer(input_without_examples, padding=False)["input_ids"])

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

            while len(positive_examples) > 0 or len(negative_examples) > 0:
                if len(positive_examples) > 0:
                    next_positive_example = positive_examples.pop()
                    next_positive_example_str = next_positive_example[0]
                    next_positive_example_length = next_positive_example[1]
                    if current_input_length + next_positive_example_length > self.max_input_length:
                        break
                    else:
                        current_input_length += next_positive_example_length
                        selected_positive_examples.append(next_positive_example_str)
                if len(negative_examples) > 0:
                    next_negative_example = negative_examples.pop()
                    next_negative_example_str = next_negative_example[0]
                    next_negative_example_length = next_negative_example[1]
                    if current_input_length + next_negative_example_length > self.max_input_length:
                        break
                    else:
                        current_input_length += next_negative_example_length
                        selected_negative_examples.append(next_negative_example_str)

            # Format the final input with the selected examples.
            positive_examples_str = "\n".join(selected_positive_examples)
            negative_examples_str = "\n".join(selected_negative_examples)

            input_with_examples = template.format(
                title=task_data["Title"],
                definition=task_data["Definition"],
                prompt=task_data["Prompt"],
                avoid=task_data["Things to Avoid"],
                emphasis=task_data["Emphasis & Caution"],
                negative_examples=negative_examples_str.strip(),
                positive_examples=positive_examples_str.strip(),
                input=sample["input"],
            )

            # Randomly select a reference output.
            output = random.choice(sample["output"])

            self.sample_inputs.append(input_with_examples)
            self.sample_outputs.append(output)

    def __len__(self):
        return len(self.sample_inputs)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.sample_inputs[idx], padding=self.padding, max_length=self.max_input_length, truncation=True, return_tensors="pt")
        outputs = self.tokenizer(self.sample_outputs[idx], padding=self.padding, max_length=self.max_output_length, truncation=True, return_tensors="pt")
        labels = outputs["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = HF_LOSS_IGNORE_TOKEN_ID

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            # Hugging Face's Seq2Seq models needs a labels argument, and should be able to generate decoder attention mask automatically.
            "labels": labels.flatten(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./dev_config.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        config = parsed = DotMap(json.load(f))

    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
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
    )
    print(f"Finished loading datasets")

    for i, batch in enumerate(val_dataset):
        print("Sample Index:", i)
        input_str = tokenizer.decode(batch["input_ids"])
        output_ids = [token_id if token_id != HF_LOSS_IGNORE_TOKEN_ID else tokenizer.pad_token_id for token_id in batch["labels"]]
        print("Input:\n", tokenizer.decode(batch["input_ids"]))
        print("Output:\n", tokenizer.decode(output_ids))
        num_non_padded_tokens = len([token_id for token_id in batch["input_ids"] if token_id != tokenizer.pad_token_id])
        print("Number of non-padded tokens in inputs:", num_non_padded_tokens)
        num_non_padded_tokens = len([token_id for token_id in output_ids if token_id != tokenizer.pad_token_id])
        print("Number of non-padded tokens in outputs:", num_non_padded_tokens)

        if i > 1000:
            break
