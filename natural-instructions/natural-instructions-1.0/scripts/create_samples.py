import argparse
import json


# we try our best to follow the context template in the paper
# see https://arxiv.org/abs/2104.08773, Appendix B
def get_context(data):
    context = f"""Title: {data['Title']}

Definition: {data['Definition']}

Prompt: {data['Prompt']}

Things to Avoid: {data['Things to Avoid']}

Emphasis & Caution: {data['Emphasis & Caution']}

"""

    context += "Negative Examples:"
    # check if negative examples are present
    if "Negative Examples" not in data['Examples']:
        context += " -\n"
    else:
        context += "\n"
        for i in range(len(data['Examples']['Negative Examples'])):
            # check if reason is present in the example
            if "reason" not in data['Examples']['Negative Examples'][i]:
                # following in the format in the paper
                # see https://arxiv.org/abs/2104.08773
                reason = "-"
            else:
                reason = data['Examples']['Negative Examples'][i]['reason']
            # check if suggestion is present in the example
            if "suggestion" not in data['Examples']['Negative Examples'][i]:
                # following in the format in the paper
                # see https://arxiv.org/abs/2104.08773
                suggestion = "-"
            else:
                suggestion = data['Examples']['Negative Examples'][i]['suggestion']
            context += f"""Negative Example {i+1} -
    Input: {data['Examples']['Negative Examples'][i]['input']}
    Output: {data['Examples']['Negative Examples'][i]['output']}
    Reason: {reason}
    Suggestion: {suggestion}

"""

    context += """Positive Examples:
"""

    for i in range(len(data['Examples']['Positive Examples'])):
        # check if reason is present in the example
        if "reason" not in data['Examples']['Positive Examples'][i]:
            # following in the format in the paper
            # see https://arxiv.org/abs/2104.08773
            reason = "-"
        else:
            reason = data['Examples']['Positive Examples'][i]['reason']
        # check if suggestion is present in the example
        if "suggestion" not in data['Examples']['Positive Examples'][i]:
            # following in the format in the paper
            # see https://arxiv.org/abs/2104.08773
            suggestion = "-"
        else:
            suggestion = data['Examples']['Positive Examples'][i]['suggestion']
        context += f"""Positive Example {i+1} -
    Input: {data['Examples']['Positive Examples'][i]['input']}
    Output: {data['Examples']['Positive Examples'][i]['output']}
    Reason: {reason}
    Suggestion: {suggestion}

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
    # task_name = args.task_path.split("\\")[-1].split(".")[0]

    # get the context with few-shot examples
    context = get_context(data)
    print(context)

    # TODO: append input to the context
    # key is data['Instances'][i]['Input']


if __name__ == "__main__":
    main()
