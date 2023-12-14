# Demystifing The In-Context Contrastive Learning Paradox
## For All Experiments Involving Natural Instructions V1
- Root Folder: `cd ./natural-instructions`
- All training data are included in the repository.
- Install Dependencies: `cd ./natural-instructions; pip3 install -r requirements.txt`
- Running BART Training: `cd ./natural-instructions; python3 train.py --config ./configs/target_config.json`
- Running Mistral-7B Training with LoRA: `cd ./natural-instructions; python3 train_llm_lora.py --config ./configs/target_llm_config.json`
- Perform text generation with: `cd ./natural-instructions; python3 generate.py --config ./configs/target_config.json` The config file is the same one used for training.
- Perform scoring with: `cd ./natural-instructions; python3 evaluate_res.py --results ./generation_results/target_generation_results.json`
- All experiment configs are saved in the `./natural-instructions/configs/` folder and therefore fully reproducible. Loss data will be saved in `./natural-instructions/logs/` as TensorBoard log files.

## For All Experiments Involving GYAFC
- Root Folder: `cd ./text-style-transfer`
- Reformat GYAFC data: `cd ./text-style-transfer; python3 reformat_data.py --data [subdataset]` where `[subdataset]` is one of `Entertainment_Music`, `Family_Relationships`.
- Running Inference with GPT-3.5: `cd ./text-style-transfer; python3 gpt3_inference.py --task [task_name] --data [subdataset] --pos_num [number_of_positive_samples] --neg_num [number_of_negative_samples]` where `[task_name]` is one of `formal-2-informal`, `informal-2-formal`, `[subdataset]` is one of `Entertainment_Music`, `Family_Relationships`. Use the `--reverse` flag to reverse order of the positive and negative samples. Use the `--shuffle` flag to randomly shuffle the positive and negative samples.
- Evaluate Inference Results: `cd ./text-style-transfer; python3 evaluate_res.py --task [task_name] --data [subdataset]` where `[task_name]` is one of `formal-2-informal`, `informal-2-formal`, `[subdataset]` is one of `Entertainment_Music`, `Family_Relationships`.