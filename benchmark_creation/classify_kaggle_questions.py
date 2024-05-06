import os
import json
import pandas as pd
import ast

from serve.utils_llm import get_llm_output

df = pd.read_csv("data/arena/kaggle.csv")
print(len(df))
df = df.sample(5000, random_state=42)
# convert to records
df = df.to_dict("records")

new_prompt = """**Task: Classify the following questions into the appropriate categories and subcategories based on their content.**

### Categories & Subcategories:
1. **Knowledge-intensive**
   - Humanities and social sciences
   - Natural sciences
   - Engineering and technology
   - Entertainment
   - Other
2. **Creative writing**
   - Role playing
   - Brainstorming
   - Poetry
   - Literary prose
   - Non-literary prose
3. **Input-based**
   - Data processing
   - Reading comprehension
   - Classification
   - Extraction
   - Summarization
   - Rewriting
   - Translation
4. **Reasoning**
   - Maths
   - Commonsense and logical reasoning
   - Instruction following
5. **Coding**
   - N/A

### Question: {question}

Plese provide the category and subcategory for the question in the following format:
- Category: {{category}}
- Subcategory: {{subcategory}}"""


import re

def parse_category_subcategory(text):
    # Regular expression to match the category and subcategory
    pattern = r'- Category: (.+?)\n- Subcategory: (.+)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Extract the category and subcategory
        category = match.group(1)
        subcategory = match.group(2)
        return category, subcategory
    else:
        return None, None

        
from tqdm import tqdm
logs = []
for i in tqdm(range(len(df))):
    example_question = df[i]['question']
    print(f"Question: {example_question}")
    ans = get_llm_output(new_prompt.format(question=example_question), "llama-3-70b")
    category, subcategory = parse_category_subcategory(ans)
    logs.append({"question": example_question, "response": ans, "category": category, "subcategory": subcategory})

# save logs to json
with open("classify_questions_kaggle.json", "w") as f:
    json.dump(logs, f, indent=4)



import wandb
run = wandb.init(project="KaggleVibeCheck", entity="lisabdunlap")
# log logs table
wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
run.finish()