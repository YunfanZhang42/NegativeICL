import argparse
import json
from evaluate import load

"""
RES = [ 
        "bart-large-all-tasks-no-few-shot.json",
        "bart-large-all-tasks-pos-max.json",
        "gpt-3.5-pos-2.json",
        "gpt-3.5-pos-2-neg-2.json",
        "bart-large-no-few-shot.json",
        "bart-large-pos-2.json",
        "bart-large-pos-4.json",
        "bart-large-pos-2-neg-2.json",
        "bart-large-neg-2.json",
    ]
"""

RES = [
    "bart-large-neg-2.json",
    "gpt-3.5-neg-2.json",
    ]

# calculate ROUGE score
def calculate_ROUGE(reference, candidate):
    metric = load('rouge')
    result = metric.compute(references=reference, predictions=candidate)
    return result


# calculate BERTScore
# for our experiments, the natural language descriptions are in English
# but it can be customized to other natural languages
def calculate_BERTScore(reference, candidate):
    metric = load("bertscore")
    result = metric.compute(references=reference, predictions=candidate, lang="en")
    return result

# calculate BLEU score
def calculate_BLEU(reference, candidate):
    metric = load("bleu")
    result = metric.compute(references=reference, predictions=candidate)
    return result


# calculate chrF score
def calculate_chrF(reference, candidate):
    metric = load("chrf")
    result = metric.compute(references=reference, predictions=candidate)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--results", type=str, default="./generation_results/gpt-3.5-pos-2-neg-2.json", help="Path to config file")
    args = parser.parse_args()

    for res in RES:
        input(f"Press Enter to continue to {res}...")
        print("************************************************************")
        print(f"Results for {res}")
        with open(f"./generation_results/{res}", "r") as f:
            results = json.load(f)

        total_score = {
            "bleu": 0,
            "rouge1": 0,
            "rouge2": 0,
            "rougeL": 0,
            "chrf": 0,
            "bertscore": 0,
        }
        subtask_022 = {
            "bleu": 0,
            "rouge1": 0,
            "rouge2": 0,
            "rougeL": 0,
            "chrf": 0,
            "bertscore": 0,
        }

        for key, values in results.items():
            print("--------------------------------------------")
            print(key)
            print("--------------------------------------------")
            # calcuate BLEU
            bleu = calculate_BLEU(values["gt"], values["pred"])["bleu"] * 100
            # calculate ROUGE
            rouge_res = calculate_ROUGE(values["gt"], values["pred"])
            rouge_1 = rouge_res["rouge1"] * 100
            rouge_2 = rouge_res["rouge2"] * 100
            rouge_L = rouge_res["rougeL"] * 100
            # calculate chrF
            chrf = calculate_chrF(values["gt"], values["pred"])["score"]
            # calculate BERTScore
            # bertscore = calculate_BERTScore(values["gt"], values["pred"])["f1"][0]
            bertscore = 0
            
            # add to total score
            total_score["bleu"] += bleu
            total_score["rouge1"] += rouge_1
            total_score["rouge2"] += rouge_2
            total_score["rougeL"] += rouge_L
            total_score["chrf"] += chrf
            total_score["bertscore"] += bertscore
            # add to subtask 022 score
            if key.startswith("subtask022"):
                print("This is subtask022!!!!")
                subtask_022["bleu"] = bleu
                subtask_022["rouge1"] = rouge_1
                subtask_022["rouge2"] = rouge_2
                subtask_022["rougeL"] = rouge_L
                subtask_022["chrf"] = chrf
                subtask_022["bertscore"] = bertscore
            
            # print results
            print("--------------------------------------------")
            print(f"BLEU: {bleu}")
            print(f"ROUGE-1: {rouge_1}")
            print(f"ROUGE-2: {rouge_2}")
            print(f"ROUGE-L: {rouge_L}")
            print(f"chrF: {chrf}")
            # print(f"BERTScore: {bertscore}")
            print("--------------------------------------------")

        # calculate average score
        avg_score = {
            "bleu": total_score["bleu"] / len(results),
            "rouge1": total_score["rouge1"] / len(results),
            "rouge2": total_score["rouge2"] / len(results),
            "rougeL": total_score["rougeL"] / len(results),
            "chrf": total_score["chrf"] / len(results),
            # "bertscore": total_score["bertscore"] / len(results),
        }
        # calculate average score without subtask 022
        avg_score_without_022 = {
            "bleu": (total_score["bleu"] - subtask_022["bleu"]) / (len(results) - 1),
            "rouge1": (total_score["rouge1"] - subtask_022["rouge1"]) / (len(results) - 1),
            "rouge2": (total_score["rouge2"] - subtask_022["rouge2"]) / (len(results) - 1),
            "rougeL": (total_score["rougeL"] - subtask_022["rougeL"]) / (len(results) - 1),
            "chrf": (total_score["chrf"] - subtask_022["chrf"]) / (len(results) - 1),
            # "bertscore": (total_score["bertscore"] - subtask_022["bertscore"]) / (len(results) - 1),
        }
        print("============================================================")
        print(f"Average BLEU: {avg_score['bleu']}")
        print(f"Average ROUGE-1: {avg_score['rouge1']}")
        print(f"Average ROUGE-2: {avg_score['rouge2']}")
        print(f"Average ROUGE-L: {avg_score['rougeL']}")
        print(f"Average chrF: {avg_score['chrf']}")
        # print(f"Average BERTScore: {avg_score['bertscore']}")
        print("============================================================")
        print(f"Average BLEU without subtask 022: {avg_score_without_022['bleu']}")
        print(f"Average ROUGE-1 without subtask 022: {avg_score_without_022['rouge1']}")
        print(f"Average ROUGE-2 without subtask 022: {avg_score_without_022['rouge2']}")
        print(f"Average ROUGE-L without subtask 022: {avg_score_without_022['rougeL']}")
        print(f"Average chrF without subtask 022: {avg_score_without_022['chrf']}")
        # print(f"Average BERTScore without subtask 022: {avg_score_without_022['bertscore']}") 
        print("************************************************************")
