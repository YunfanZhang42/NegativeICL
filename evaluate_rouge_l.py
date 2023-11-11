import argparse
import json
import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--results", type=str, default="./generation_results/gpt-3.5-pos-2-neg-2.json", help="Path to config file")
    args = parser.parse_args()

    results = json.load(open(args.results, "r"))

    rouge = evaluate.load('rouge')

    total_score = 0
    for key, values in results.items():
        print(key)
        score = rouge.compute(predictions=values["pred"], references=values["gt"], rouge_types=["rougeL"])
        total_score += score["rougeL"]
        print(f"Rouge-L: {score['rougeL']}")
    
    avg_score = total_score / len(results)
    print(f"Average Rouge-L: {avg_score}")
