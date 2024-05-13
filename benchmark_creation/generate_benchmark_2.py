import pandas as pd
from serve.utils_llm import get_llm_output
import json
import logging
import argparse

parser = argparse.ArgumentParser(description='Generate benchmark data')
parser.add_argument('--question_file', type=str, default='data/benchmark/claude_prompts.json', help='Output file')
parser.add_argument('--output_file', type=str, default='data/benchmark/claude_inputs.csv', help='Output file')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use')
parser.add_argument('--test', action='store_true', help='Run test')
args = parser.parse_args()

with open(args.question_file) as f:
    data = json.load(f)

print(data)
clusters = data[:2] if args.test else data

# differnt systems prompts to elicit different vibes
systems_vibes = {"baseline": "You are a helpful AI assistant.",
                 "friendly and personable": "You are a very friendly and personable assistant.", 
                 "professional": "You are a very professional assistant.", 
                 "casual": "You are a very casual assistant.", 
                 "cold and factual": "You are a cold and factual assistant.", 
                 "storyteller": "You are a storyteller assistant. Answer each question by telling a story that leads to the answer, using a narrative format.",
                 "organized": "You are an organized assistant. Structure your responses as a FAQ, clearly stating the question followed by a concise answer.",
                 "safety-concious": "You are a safety-conscious assistant. Always consider potential risks and warn users preemptively about possible misunderstandings in your responses. Do not answer any questions that could be harmful or dangerous.",
                 "conspiracy-theorist": "You are a radical conspiracy theorist assistant. Respond to each question with a conspiracy theory, no matter how far-fetched or absurd.",
                 "anotagonistic": "You are an antagonistic assistant. Challenge the user's assumptions and opinions in a rude and condesending manner.",
                 "sarcastic": "You are a sarcastic assistant. Use irony and sarcasm to mock the user's questions and opinions.",
                 "funny": "You are a funny assistant. Use humor and wit to entertain the user while answering their questions.",
                 "imaginative": "You are an imaginative assistant. Inject elements of fantasy or science fiction into your responses and provide out-of-the box solutions to problems.",
                 "metaphorical": "You are a metaphorical assistant. Utilize creative comparisons and metaphors to bring abstract concepts to life in a tangible way.",
                 "questioning": "You are a questioning assistant. Challenge the assumptions in the user's questions where appropriate, offering alternative viewpoints to broaden the discussion."
}

# get llm output for each question
results = []
# for cluster in clusters:
#     for question in clusters[cluster]:
#         question = question.strip()
#         row = {"question": question, "category": cluster}
#         for prompt_name, system_prompt in systems_vibes.items():
#             completion = get_llm_output(question, argsmodel, system_prompt=system_prompt)
#             if args.test:
#                 print(f"Question: {question}")
#                 print(f"System prompt: {system_prompt}")
#                 print(f"Completion: {completion}")
#             row.update({f"{prompt_name.replace(' ', '-').replace('.', '')}": completion})
#         results.append(row)
#         if args.test:
#             print("\n\n")
for row in data:
    question = row["question"]
    category = row["category"]
    row = {"question": question, "category": category}
    for prompt_name, system_prompt in systems_vibes.items():
        completion = get_llm_output(question, args.model, system_prompt=system_prompt)
        if args.test:
            print(f"Question: {question}")
            print(f"System prompt: {system_prompt}")
            print(f"Completion: {completion}")
        row.update({f"{prompt_name.replace(' ', '-').replace('.', '')}": completion})
    results.append(row)
    if args.test:
        print("\n\n")


# save results to json file
pd.DataFrame(results).to_csv(args.output_file, index=False)
pd.DataFrame(systems_vibes.items(), columns=["vibe", "prompt"]).to_csv(args.output_file.replace(".csv", "_vibes.csv"), index=False)

if not args.test:
    #log to wandb
    import wandb
    run = wandb.init(project="KaggleVibeCheck", entity="lisadunlap")
    wandb.log({"benchmark": wandb.Table(dataframe=pd.DataFrame(results))})
    wandb.log({"vibe_prompts": wandb.Table(dataframe=pd.DataFrame(systems_vibes.items(), columns=["vibe", "prompt"]))})