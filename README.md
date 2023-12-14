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

