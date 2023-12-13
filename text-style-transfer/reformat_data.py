import argparse
import json
from tqdm import tqdm


if __name__ == "__main__":
    # python format_data.py --data Entertainment_Music
    parser = argparse.ArgumentParser(description="reformat GYAFC data")
    parser.add_argument("--data", type=str, default="Entertainment_Music", help="Path to config file")
    args = parser.parse_args()
    
    # reformat test data
    splits = ["test", "tune"]
    types = ["formal", "informal"]
    refs = ["0", "1", "2", "3"]
    
    for split in splits:
        for type in types:
            # reading problem file
            with open(f"./GYAFC_Corpus/{args.data}/{split}/{type}", "r") as problem_file:
                problem_lines = problem_file.readlines()
            
            # reading gound truth files
            gt_lines_all = []
            for ref in refs:
                if type == "formal":
                    with open(f"./GYAFC_Corpus/{args.data}/{split}/informal.ref{ref}", "r") as gt_file:
                        gt_lines_all.append(gt_file.readlines())
                else:
                    with open(f"./GYAFC_Corpus/{args.data}/{split}/formal.ref{ref}", "r") as gt_file:
                        gt_lines_all.append(gt_file.readlines())
                        
            # create writing path
            if type == "formal":
                path = f"./GYAFC_Corpus/{args.data}/{split}/formal-2-informal.jsonl"
            else:
                path = f"./GYAFC_Corpus/{args.data}/{split}/informal-2-formal.jsonl"
                
            # writing reformatted file
            with open(path, "w") as reformatted_file:
                for i in tqdm(range(len(problem_lines))):
                    # create reformatted file template
                    task = {
                        "problem": "",
                        "gts": [],
                    }
                    # writing problem description
                    task["problem"] = problem_lines[i].strip()
                    # writing ground truth
                    gts = []
                    for gt_lines in gt_lines_all:
                        gts.append(gt_lines[i].strip())
                    # print(gts)
                    task["gts"] = gts
                    # dump one task to reformatted file
                    # print(task)
                    # input("Press Enter to continue...")
                    reformatted_file.write(json.dumps(task) + "\n")
                    reformatted_file.flush()
