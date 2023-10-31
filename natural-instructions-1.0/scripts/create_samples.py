import argparse
import json


def get_context(data):
    context = f"""Title: {data['Title']}

Definition: {data['Definition']}

Emphasis & Caution: {data['Emphasis & Caution']}

Things to Avoid: {data['Things to Avoid']}

Positive Examples:

"""

    for i in range(len(data['Examples']['Positive Examples'])):
        # checking if suggestions is present in the example
        if "suggestion" not in data['Examples']['Positive Examples'][i]:
            # following in the format in the paper
            # see https://arxiv.org/abs/2104.08773
            suggestion = "-"
        else:
            suggestion = data['Examples']['Positive Examples'][i]['suggestion']
        context += f"""Input: {data['Examples']['Positive Examples'][i]['input']}
Output: {data['Examples']['Positive Examples'][i]['output']}
Reason: {data['Examples']['Positive Examples'][i]['reason']}
Suggestion: {suggestion}

"""

    context += """Negative Examples:
"""

    for i in range(len(data['Examples']['Negative Examples'])):
        # checking if suggestions is present in the example
        if "suggestion" not in data['Examples']['Negative Examples'][i]:
            # following in the format in the paper
            # see https://arxiv.org/abs/2104.08773
            suggestion = "-"
        else:
            suggestion = data['Examples']['Negative Examples'][i]['suggestion']
        context += f"""Input: {data['Examples']['Negative Examples'][i]['input']}
Output: {data['Examples']['Negative Examples'][i]['output']}
Reason: {data['Examples']['Negative Examples'][i]['reason']}
Suggestion: {suggestion}

"""

    context += f"""Prompt: {data['Prompt']}

"""

    return context
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_path", type=str, required=True)
    args = parser.parse_args()

    # read in data    
    with open(args.task_path, "r", encoding="utf-8") as reader:
        data = reader.read()
    data = json.loads(data)

    # get the context with few-shot examples
    context = get_context(data)
    print(context)

    # TODO: append input to the context
    # key is data['Instances'][i]['Input']


if __name__ == "__main__":
    main()
