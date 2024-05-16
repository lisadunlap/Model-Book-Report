import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree

def expand_dataframe_with_axes(df):
  unique_axes = df['axis'].unique()
  existing_pairs = set(zip(df['question'], df['axis']))
  new_rows = []
  for question in df['question'].unique():
      for axis in unique_axes:
          if (question, axis) not in existing_pairs:
              new_row = {
                  'question': question,
                  'axis': axis,
                  'avg_scores': [0, 0],
                  'avg_diff_scores': [0, 0]
              }
              new_rows.append(new_row)
  new_rows_df = pd.DataFrame(new_rows)
  return pd.concat([df, new_rows_df], ignore_index=True)

def prepare_data_for_decision_tree(df, models):
  df = df.copy()
  results_short = df[['question', 'topic', 'axis', 'avg_scores', 'avg_diff_scores']]
  df = expand_dataframe_with_axes(results_short)

  # Create separate rows for each model in the models list
  expanded_rows = []
  for i, model in enumerate(models):
      model_rows = df.copy()
      model_rows['label'] = model
      model_rows['avg_diff_scores'] = model_rows['avg_diff_scores'].apply(lambda x: x[i])  # Select the element for the current model
      expanded_rows.append(model_rows)

  # Concatenate all model rows
  expanded_df = pd.concat(expanded_rows, ignore_index=True)

  # Pivot the data to create feature columns for each axis
  pivot_df = expanded_df.pivot_table(index=['question', 'label'], columns='axis', values='avg_diff_scores', fill_value=0).reset_index()

  # Prepare features and target
  X = pivot_df.drop(columns=['question', 'label'])
  y = pivot_df['label']

  return X, y

def train_decision_tree(results, test_results, models):
  X_train, y_train = prepare_data_for_decision_tree(results, models)
  X_test, y_test = prepare_data_for_decision_tree(test_results, models)
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_train_pred = clf.predict(X_train)

  print(f"Training a decsion tree with {len(models)} models on {len(X_train)} samples")
  print("Train Classification Report:")
  print(classification_report(y_train, y_train_pred))
  print("Test Classification Report:")
  print(classification_report(y_test, y_pred))
   
  clf = LogisticRegression(random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_train_pred = clf.predict(X_train)

  print(f"Training a logistic regression with {len(models)} models on {len(X_train)} samples")
  print("Train Classification Report:")
  print(classification_report(y_train, y_train_pred))
  print("Test Classification Report:")
  print(classification_report(y_test, y_pred))