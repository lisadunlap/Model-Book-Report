from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# Function to get embeddings from a list of text
def get_embeddings(text_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    return np.vstack(embeddings)

# Function to train classifier
def train_classifier(text_list, labels):
    embeddings = get_embeddings(text_list)
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(embeddings, labels)
    return classifier

import ast
import pandas as pd
df = pd.read_csv('data/chatbot_arena_conversations_1_turn.csv')
df['conversation_a'] = df['conversation_a'].apply(ast.literal_eval)
df['conversation_b'] = df['conversation_b'].apply(ast.literal_eval)
df['answer_a'] = df['conversation_a'].apply(lambda x: x[1]['content'])
df['answer_b'] = df['conversation_b'].apply(lambda x: x[1]['content'])
model1 = "vicuna-13b"
model2 = "koala-13b"
two_model_df = df[((df['model_a'] == model1) & (df['model_b'] == model2)) | 
                ((df['model_a'] == model2) & (df['model_b'] == model1))]


# Create two separate dataframes, one for each model and its corresponding conversation
df_model_a = two_model_df[['question_id', 'question', 'model_a', 'winner', 'answer_a']].copy()
df_model_b = two_model_df[['question_id', 'question', 'model_b', 'winner', 'answer_b']].copy()

# Rename the columns to have consistent names
df_model_a.rename(columns={'model_a': 'model', 'answer_a': 'answer'}, inplace=True)
df_model_b.rename(columns={'model_b': 'model', 'answer_b': 'answer'}, inplace=True)

# Concatenate the two dataframes
df_final = pd.concat([df_model_a, df_model_b], ignore_index=True)
df = df_final[df_final['winner'] != 'tie (bothbad)']
train_df = df[:1000]
test_df = df[1000:]
# shuffle the data
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df['label'] = train_df['model'].apply(lambda x: 1 if x == 'vicuna-13b' else 0)
# Example usage
text_list = train_df['answer'].tolist() # Replace with your list of strings
labels = train_df['label'].tolist()  # Replace with your labels

# Train the classifier
classifier = train_classifier(text_list, labels)

# To predict labels for new texts
new_texts = test_df['answer'].tolist() # Replace with new texts
new_embeddings = get_embeddings(new_texts)
predictions = classifier.predict(new_embeddings)
print(predictions)

test_labels = test_df['model'].apply(lambda x: 1 if x == 'vicuna-13b' else 0).tolist()
print(predictions == test_labels)
print(sum(predictions == test_labels) / len(predictions))
