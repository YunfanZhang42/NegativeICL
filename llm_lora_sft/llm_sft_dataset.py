import random
import numpy as np

import pandas as pd
from torch.utils.data import Dataset


class LLMSFTDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        prompt_template_path,
        tokenizer,
        max_length=768,
    ):
        self.df = pd.read_parquet(dataset_path)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read().strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = self.df.iloc[idx]["input"]
        target = self.df.iloc[idx]["target"]

        text = self.prompt_template.format(
            input=input,
            target="",
        )

        text_encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False, return_tensors="np", add_special_tokens=False
        )

        text_target_encoded = self.tokenizer(
            target, truncation=True, max_length=self.max_length, padding=False, return_tensors="np", add_special_tokens=False
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

        # Truncate the combined encoding to max_length by removing tokens from the left side
        input_ids = input_ids[-self.max_length :]
        attention_mask = attention_mask[-self.max_length :]

        # Pad the combined encoding to max_length, on the left side, with padding token
        input_ids = np.pad(input_ids, (self.max_length - len(input_ids), 0), constant_values=self.tokenizer.pad_token_id)
        attention_mask = np.pad(attention_mask, (self.max_length - len(attention_mask), 0), constant_values=0)

        # Now, mark the last len(text_target_input_ids) tokens as labels, and the rest as -100
        labels = np.full(self.max_length, -100)
        labels[-len(text_target_input_ids) :] = text_target_input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input": input,
            "target": target,
        }
