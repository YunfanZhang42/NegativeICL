import argparse
import json
from evaluate import load


FORMAL_2_INFORMAL = [
    #"formal-2-informal_gpt-3.5_pos-0_neg-0",
    #"formal-2-informal_gpt-3.5_pos-1_neg-0",
    #"formal-2-informal_gpt-3.5_pos-2_neg-0",
    #"formal-2-informal_gpt-3.5_pos-4_neg-0",
    #"formal-2-informal_gpt-3.5_pos-1_neg-3",
    #"formal-2-informal_gpt-3.5_pos-1_neg-3_reversed",
    #"formal-2-informal_gpt-3.5_pos-2_neg-2",
    #"formal-2-informal_gpt-3.5_pos-2_neg-2_reversed",
    #"formal-2-informal_gpt-3.5_pos-3_neg-1",
    #"formal-2-informal_gpt-3.5_pos-3_neg-1_reversed",
    #"formal-2-informal_gpt-3.5_pos-6_neg-0",
    #"formal-2-informal_gpt-3.5_pos-2_neg-4",
    "formal-2-informal_gpt-3.5_pos-0_neg-1",
    #"formal-2-informal_gpt-3.5_pos-0_neg-2",
    #"formal-2-informal_gpt-3.5_pos-0_neg-4",
]
INFORMAL_2_FORMAL = [
    #"informal-2-formal_gpt-3.5_pos-0_neg-0",
    #"informal-2-formal_gpt-3.5_pos-1_neg-0",
    #"informal-2-formal_gpt-3.5_pos-2_neg-0",
    #"informal-2-formal_gpt-3.5_pos-4_neg-0",
    "informal-2-formal_gpt-3.5_pos-1_neg-3",
    #"informal-2-formal_gpt-3.5_pos-2_neg-2",
    "informal-2-formal_gpt-3.5_pos-3_neg-1",
    "informal-2-formal_gpt-3.5_pos-0_neg-1",
    "informal-2-formal_gpt-3.5_pos-0_neg-2",
    "informal-2-formal_gpt-3.5_pos-0_neg-4",
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
    # python evaluate_res.py --task informal-2-formal --data Family_Relationships
    parser = argparse.ArgumentParser(description="Evaluate GYAFC dataset.")
    parser.add_argument("--task", type=str, default="formal-2-informal", help="task name")
    parser.add_argument("--data", type=str, default="Entertainment_Music", help="Path to config file")
    args = parser.parse_args()

    if args.task == "formal-2-informal":
        res_all = FORMAL_2_INFORMAL
    elif args.task == "informal-2-formal":
        res_all = INFORMAL_2_FORMAL
    else:
        raise ValueError("Invalid task name!")
    
    for res in res_all:
        input(f"Press Enter to continue to {res}...")
        print("************************************************************")
        print(f"Results for {res}")
        # read the results file
        with open(f"./GYAFC_Corpus/{args.data}/model_outputs/{res}.json", "r") as res_file:
            results = json.load(res_file)
        
        # initialize scores
        bleu = 0
        rouge1 = 0
        rouge2 = 0
        rougeL = 0
        chrf = 0
        for i in range(4):
            # get predictions
            pred = results["pred"]
            gt = results[f"gt_{i}"]
            # calculate BLEU score
            bleu += calculate_BLEU(gt, pred)["bleu"] * 100
            # calculate ROUGE score
            rouge_res = calculate_ROUGE(gt, pred)
            rouge1 += rouge_res["rouge1"] * 100
            rouge2 += rouge_res["rouge2"] * 100
            rougeL += rouge_res["rougeL"] * 100
            # calculate chrF score
            chrf += calculate_chrF(gt, pred)["score"]
            
        # average scores across 4 references
        bleu /= 4
        rouge1 /= 4
        rouge2 /= 4
        rougeL /= 4
        chrf /= 4
        
        print("************************************************************")
        print(f"BLEU: {bleu}")
        print(f"ROUGE-1: {rouge1}")
        print(f"ROUGE-2: {rouge2}")
        print(f"ROUGE-L: {rougeL}")
        print(f"chrF: {chrf}")
        print("************************************************************")
