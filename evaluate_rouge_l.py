import argparse
import json
import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--results", type=str, default="./generation_results/bart-large-pos-4.json", help="Path to config file")
    args = parser.parse_args()

    results = json.load(open(args.results, "r"))

    rouge = evaluate.load('rouge')

    for key, values in results.items():
        print(key)
        score = rouge.compute(predictions=values["pred"], references=values["gt"], rouge_types=["rougeL"])
        print(f"Rouge-L: {score['rougeL']}")
