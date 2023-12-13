# Authors: marcusm117


# Standard Library Modules
import argparse
import os
import time
import random
import json

# External Modules
import openai
from pprint import pprint
from tqdm import tqdm

# add your OpenAI API key here
openai.api_key = "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"


# system prompt settings
FORMAL_2_INFORMAL_ZERO_SHOT = [
    {   # Instructions
        "role": "system",
        "content": "Please convert a sentence in formal style to an informal style. "
        + "Note that you should keep the meaning of the sentence unchanged.\n\n"
    }
]
FORMAL_2_INFORMAL = [
    {  # Instructions
        "role": "system",
        "content": "Please convert a sentence in formal style to an informal style. "
        + "Note that you should keep the meaning of the sentence unchanged. "
        + "Please follow the positive examples and avoid the negative examples shown below.\n\n",
    }
]
INFORMAL_2_FORMAL_ZERO_SHOT = [
    {   # Instructions
        "role": "system",
        "content": "Please convert a sentence in informal style to an formal style. "
        + "Note that you should keep the meaning of the sentence unchanged.\n\n"
    }
]
INFORMAL_2_FORMAL = [
    {  # Instructions
        "role": "system",
        "content": "Please convert a sentence in informal style to an formal style. "
        + "Note that you should keep the meaning of the sentence unchanged. "
        + "Please follow the positive examples and avoid the negative examples shown below.\n\n",
    }
]


def generate_prompt_quantity(system_prompt, pos_example_bank, neg_example_bank, pos_num, neg_num, reverse=False, shuffle=False):
    # initialize the prompt
    messages = []
    
    # randomly select positive examples
    pos_prompt = []
    for i in range(pos_num):
        bank_size = len(pos_example_bank)
        pos_ex = json.loads(pos_example_bank[random.randint(0, bank_size-1)])
        pos_prompt.append(
            [
                {
                    "role": "user",
                    "content": f"Positive Example {i+1}\nInput: {pos_ex['problem']}\n\n"
                },
                {
                    "role": "assistant",
                    "content": f"Output:{pos_ex['gts'][0]}\n\n"
                }
            ]
        )

    # randomly select negative examples
    neg_prompt = []
    for i in range(neg_num):
        bank_size = len(neg_example_bank)
        neg_ex = json.loads(neg_example_bank[random.randint(0, bank_size-1)])
        neg_prompt.append(
            [
                {
                    "role": "user",
                    "content": f"Negative Example {i+1}:\nInput: {neg_ex['gts'][0]}\n\n"
                },
                {
                    "role": "assistant",
                    "content": f"Output: {neg_ex['gts'][1]}\n\n"
                }
            ]
        )
    
    if reverse:
        messages = system_prompt + sum(neg_prompt, []) + sum(pos_prompt, [])
    elif shuffle:
        example_prompt = pos_prompt + neg_prompt
        random.shuffle(example_prompt)
        messages = system_prompt + sum(example_prompt, [])
    else:
        messages = system_prompt + sum(pos_prompt, []) + sum(neg_prompt, [])

    # print(messages)
    return messages


# get completion from an OpenAI chat model
def get_openai_chat(
    prompt,
    user_input,
    model="gpt-3.5-turbo-1106",
    temperature=0,
    max_tokens=128,
    seed=0,
):
    # select the correct in-context learning prompt based on the task
    messages = prompt + [{"role": "user", "content": user_input}]
    # print(messages)

    # get response from OpenAI
    while True:
        try:
            response = openai.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                messages=messages,
            )
            response_content = response.choices[0].message.content.strip()
            # if the API is unstable, consider sleeping for a short period of time after each request
            # time.sleep(0.2)
            return response_content

        # when encounter RateLimit or Connection Error, sleep for 5 or specified seconds and try again
        except Exception as error:
            print(f"Rate Limit or Connection Error. Sleeping for 5 seconds ...")
            print(f"Error: {error}")
            time.sleep(5)
            return get_openai_chat(
                prompt,
                user_input,
                model,
                temperature,
                max_tokens,
                seed,
            )


def main():
    # python gpt_inference.py --task informal-2-formal --data Family_Relationships --pos_num 2 --neg_num 2 --reverse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="formal-2-informal", help="task name")
    parser.add_argument("--data", type=str, default="Entertainment_Music", help="Path to config file")
    parser.add_argument("--pos_num", type=int, default=0, help="number of positive examples")
    parser.add_argument("--neg_num", type=int, default=0, help="number of negative examples")
    parser.add_argument("--reverse", action="store_true", help="reverse the order of positive and negative examples")
    parser.add_argument("--shuffle", action="store_true", help="randomly shuffle the order of positive and negative examples")
    args = parser.parse_args()
    
    # load example bank if needed
    pos_example_bank = []
    neg_example_bank = []
    if args.pos_num > 0 or args.neg_num > 0:
        if args.task == "formal-2-informal":
            if args.pos_num > 0:
                pos_example_bank = open(f"./GYAFC_Corpus/{args.data}/tune/formal-2-informal.jsonl", "r").readlines()
            if args.neg_num > 0:
                neg_example_bank = open(f"./GYAFC_Corpus/{args.data}/tune/informal-2-formal.jsonl", "r").readlines()
        elif args.task == "informal-2-formal":
            if args.pos_num > 0:
                pos_example_bank = open(f"./GYAFC_Corpus/{args.data}/tune/informal-2-formal.jsonl", "r").readlines()
            if args.neg_num > 0:
                neg_example_bank = open(f"./GYAFC_Corpus/{args.data}/tune/formal-2-informal.jsonl", "r").readlines()
        else:
            raise ValueError("Invalid system prompt!")
    
    # load test data
    with open(f"./GYAFC_Corpus/{args.data}/test/{args.task}.jsonl", "r") as test_file:
        test_data = test_file.readlines()
    
    # write test data to file
    if args.reverse:
        outpout_path = f"./GYAFC_Corpus/{args.data}/model_outputs/{args.task}_gpt-3.5_pos-{args.pos_num}_neg-{args.neg_num}_reversed.json"
    elif args.shuffle:
        outpout_path = f"./GYAFC_Corpus/{args.data}/model_outputs/{args.task}_gpt-3.5_pos-{args.pos_num}_neg-{args.neg_num}_shuffle.json"
    else:
        outpout_path = f"./GYAFC_Corpus/{args.data}/model_outputs/{args.task}_gpt-3.5_pos-{args.pos_num}_neg-{args.neg_num}.json"
    
    with open(outpout_path, "w") as res_file:
        # initialize results json
        res = {
            "task": f"{args.data}_{args.task}",
            "problem": [],
            "pred": [],
            "gt_0": [],
            "gt_1": [],
            "gt_2": [],
            "gt_3": [],
        }
        
        # generate results for each test data task
        for i in tqdm(range(len(test_data))):
            # generate prompt for each test data task
            if args.pos_num <= 0 and args.neg_num <= 0:
                if args.task == "formal-2-informal":
                    prompt = FORMAL_2_INFORMAL_ZERO_SHOT
                elif args.task == "informal-2-formal":
                    prompt = INFORMAL_2_FORMAL_ZERO_SHOT
                else:
                    raise ValueError("Invalid task name!")
            else:
                if args.task == "formal-2-informal":
                    prompt = generate_prompt_quantity(FORMAL_2_INFORMAL, pos_example_bank, neg_example_bank, args.pos_num, args.neg_num, args.reverse, args.shuffle)
                elif args.task == "informal-2-formal":
                    prompt = generate_prompt_quantity(INFORMAL_2_FORMAL, pos_example_bank, neg_example_bank, args.pos_num, args.neg_num, args.reverse, args.shuffle)
                else:
                    raise ValueError("Invalid task name!")
            
            print(prompt)
            
            # get input from test data
            test_task = json.loads(test_data[i])
            input_cotent = test_task["problem"]
            res["problem"].append(input_cotent)
            res["gt_0"].append(test_task["gts"][0])
            res["gt_1"].append(test_task["gts"][1])
            res["gt_2"].append(test_task["gts"][2])
            res["gt_3"].append(test_task["gts"][3])
            
            # set different input format for zero-shot and few-shot
            if args.pos_num == 0 and args.neg_num == 0:
                input = f"Input: {input_cotent}\n\n"
            else:
                input = f"Positive Example {i+1}\nInput: {input_cotent}\n\n"

            # get OpenAI generated results
            result = get_openai_chat(prompt, input)
            result = result.replace("Output:", "").replace("Informal:", "").replace("Formal:", "").strip()
            result = result.replace("\n", "")
            res["pred"].append(result)
            # print(f"Output {i}: {result}")

        # write results json to file
        res_file.write(json.dumps(res, indent=4))
        res_file.flush()

if __name__ == "__main__":
    main()
