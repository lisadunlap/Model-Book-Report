import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("./data/all_train.csv")
df['label'] = df['label'].astype('category').cat.codes
# shuffle dataframe rows
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert DataFrames to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

from transformers import AutoTokenizer

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

from transformers import AutoModelForSequenceClassification

num_labels = df['label'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

import wandb

wandb.init(project="llm_eval_text_classification", entity="lisadunlap")

from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)  # Get the index of the max logit as the prediction
    return {"accuracy": accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()

# Now, you can safely call evaluate on these tokenized datasets
train_results = trainer.evaluate(tokenized_datasets["train"])
print(f"Training Accuracy: {train_results['eval_accuracy']}")

test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Testing Accuracy: {test_results['eval_accuracy']}")