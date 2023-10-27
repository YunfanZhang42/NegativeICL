from torch.utils.data import Dataset
from datasets import load_dataset


class HFCLMDataset(Dataset):
    def __init__(self, hf_dataset_path, hf_dataset_name, tokenizer, dataset_type="train", max_length=1024):
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        dataset = load_dataset(hf_dataset_path, hf_dataset_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.dataset = tokenized_datasets[dataset_type]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"],
            "labels": self.dataset[idx]["input_ids"]
        }
