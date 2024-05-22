import pandas as pd
import numpy as np
import ast
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import wandb

from components.metrics_utils import train_decision_tree, train_individual_feature_impact

def get_avg(results, judges):
  test = results[judges].to_numpy()
  test = np.array(test.tolist(), dtype=float)
  print(np.mean(test, axis=1).shape)
  return np.mean(test, axis=1)

def get_majority(results, judges):
  test = results[judges].to_numpy()
  test = np.array(test.tolist(), dtype=float)
  # get majority vote
  majority_score, count = scipy.stats.mode(test, axis=1)
  return np.squeeze(majority_score, axis=1)

def calculate_z_scores(values):
  values = np.array(values, dtype=float)
  if len(values) < 2:
      raise ValueError("List must contain at least 2 elements")
  
  mean = np.mean(values)
  std = np.std(values)
  z_scores = (values - mean) / std if std != 0 else np.zeros_like(values)
  
  return z_scores

def get_delta_binary(values):
  values = [int(x) for x in values]
  if values == [-1, 1]:
    return [-1, 1]
  elif values == [1, -1]:
    return [1, -1]
  else:
    return [0, 0]
    
def prep_results(results, judges, models):
  if len(judges) == 2:
    results["score"] = list(get_avg(results, [f"Judge_{j}_score" for j in judges]))
  else:
    results["score"] = list(get_majority(results, [f"Judge_{j}_score" for j in judges]))

  results["delta"] = results["score"].apply(calculate_z_scores)
  results_old = results.copy()

  columns = ['axis', 'question'] + models + ['topic', 'score', 'delta', 'model']
  results["model"] = [models] * len(results)
  results = results.explode(["model", "score", "delta"])
  # drop any rows which contain nan values
  results = results.dropna()
  results = results[columns]
  return results_old, results

from scipy.stats import mannwhitneyu

def cliffs_delta(x, y):
  x = np.array(x)
  y = np.array(y)
  m, n = len(x), len(y)
  x = x.reshape(-1, 1)
  y = y.reshape(1, -1)
  diff = x - y
  return np.sum(diff > 0) - np.sum(diff < 0), m * n

def analyze_differences(results, models_col='model', score_col='score', axis_col='axis'):
  models = results[models_col].unique()
  axes = results[axis_col].unique()

  # Initialize a list to store the results
  results_list = []

  # Loop through each axis and compute the statistics
  for axis in axes:
    results_axis = results[results[axis_col] == axis]
    
    # Extract scores for each model
    score_a = results_axis[results_axis[models_col] == models[0]][score_col].values
    score_b = results_axis[results_axis[models_col] == models[1]][score_col].values
    
    # Ensure scores are numerical NumPy arrays and handle NaN values
    score_a = np.array(score_a, dtype=float)
    score_b = np.array(score_b, dtype=float)
    
    # Check and remove NaN values
    score_a = score_a[~np.isnan(score_a)]
    score_b = score_b[~np.isnan(score_b)]
    
    # Perform Mann-Whitney U test
    stat, p = mannwhitneyu(score_a, score_b)
    
    # Calculate Rank-Biserial Correlation (r)
    n1 = len(score_a)
    n2 = len(score_b)
    rbc = 1 - (2 * stat) / (n1 * n2)
    
    # Calculate Cliff's Delta (d)
    delta, n_comparisons = cliffs_delta(score_a, score_b)
    d = delta / n_comparisons
    
    # Calculate the average difference between scores
    avg_diff = np.mean(score_a) - np.mean(score_b)
    
    # Append the results to the list
    results_list.append({
        axis_col: axis,
        'avg_diff': np.round(avg_diff, 3),
        'p_value': np.round(p, 3),
        'rank_biserial_correlation': np.round(rbc, 3),
        'cliffs_delta': np.round(d, 3)
    })

  # Convert the results list to a DataFrame
  results_df = pd.DataFrame(results_list)
  return results_df

def display_metrics(metric_file, models, judge="avg"):
  metrics = pd.read_json(metric_file)
  # Judge_avg_friendly-and-personable_mean_diff
  # p_value_friendly-and-personable_cold-and-factual
  model_metrics = [f"Judge_{judge}_{model}_mean_diff" for model in models]
  p_val = [p for p in metrics.columns if "p_value" in p][0]
  kappas = [p for p in metrics.columns if "Cohen's Kappa" in p and "0" in p]
  metrics["Cohn's kappa mean"] = metrics[kappas].mean(axis=1)
  summary_metrics = metrics[["topic", "axis", "support", "Cohn's kappa mean", p_val] + model_metrics]

  columns = ['topic', 'vibe', 'support', "Cohn's kappa mean", "p-value"] + [m.replace("Judge_", "").replace("_mean_diff", "") for m in model_metrics]
  for metric in model_metrics:
        summary_metrics[metric] = summary_metrics[metric].apply(lambda x: round(x, 3))
  # round metric and pval to 3 decimal places
  summary_metrics[p_val] = summary_metrics[p_val].apply(lambda x: round(x, 3))
  summary_metrics["Cohn's kappa mean"] = summary_metrics["Cohn's kappa mean"].apply(lambda x: round(x, 3))
  summary_metrics.columns = columns
  # sort by vibe then by topic
  summary_metrics = summary_metrics.sort_values(by=['vibe', 'topic']).reset_index(drop=True)
  return summary_metrics[["vibe", "topic"] + columns[2:]]
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Post hoc analysis')
  parser.add_argument('--results_dir', type=str, help='Path to the results file')
  parser.add_argument('--tag', type=str, help='Path to the output file')
  parser.add_argument('--models', nargs='+', help='List of models')
  parser.add_argument('--no-test', action='store_true', help='Use test results')
  parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
  parser.add_argument('--train_judge', type=str, help='Path to the metric file')
  args = parser.parse_args()

  # tirn off wandb logging
  if not args.wandb:
    os.environ["WANDB_MODE"] = "dryrun"
  wandb.init(project="post-hoc-analysis", config=vars(args), name=args.results_dir.replace("pipeline_results/", ""))

  results_dir = args.results_dir
  tag = args.tag
  models = list(args.models)
  
  results = pd.read_json(f"{results_dir}/{tag}-eval-results.json")
  test_results = results if args.no_test else pd.read_json(f"{results_dir}/{tag}-test-results.json")

  topics = results["topic"].unique()
  for topic in topics:
    try:
      results = pd.read_json(f"{results_dir}/{tag}-eval-results.json")
      test_results = results if args.no_test else pd.read_json(f"{results_dir}/{tag}-test-results.json")
      if args.no_test:
        train_idxs = np.random.choice(results.index, int(0.5 * len(results)), replace=False)
        test_idxs = [idx for idx in results.index if idx not in train_idxs]
        test_results = results.loc[test_idxs]
        results = results.loc[train_idxs]
      results = results[results["topic"] == topic]
      test_results = test_results[test_results["topic"] == topic]
      print(f"Analyzing {topic} (train = {len(results)}, test = {len(test_results)})")
      print(results.axis.value_counts())
      # only keep axes which appear in both results and test_results
      results = results[results["axis"].isin(test_results["axis"])]
      test_results = test_results[test_results["axis"].isin(results["axis"])]

      judges = ["0", "1"]
      if "Judge_2" in results.columns:
        judges += ["2"]
      if "Judge_3" in results.columns:
        judges += ["3"]

      if args.train_judge:
        results_old, results = prep_results(results, [args.train_judge], models)
        test_results_old, test_results = prep_results(test_results, judges, models)
      else:
        results_old, results = prep_results(results, judges, models)
        test_results_old, test_results = prep_results(test_results, judges, models)

      fig = plt.figure(figsize=(20,20))
      sns.barplot(y="axis", x="score", hue="model", data=results)
      plt.title("Average Scores")
      plt.savefig(f"{results_dir}/{tag}-avg-scores.png", bbox_inches='tight')
      wandb.log({f"{topic} avg_scores": wandb.Image(f"{results_dir}/{tag}-avg-scores.png")})

      results_df = analyze_differences(results)

      important_axes = results_df['axis'].values
      results_df.sort_values(by="avg_diff", ascending=False)

      train = results_old[results_old["axis"].isin(important_axes)]
      test = test_results_old[test_results_old["axis"].isin(important_axes)]
      _, _, report = train_individual_feature_impact(train, test, models, metric="delta")
      features = report['logistic_regression'].keys()

      acc = pd.DataFrame({f"axis": features, "accuracy": [np.round(report['logistic_regression'][feature]['accuracy'], 3) for feature in features]})
      new_results_df = results_df.merge(acc, on="axis", how='outer')
      new_results_df.sort_values(by="accuracy", ascending=False)

      metric_file = f"{results_dir}/{tag}-eval-metrics.json"
      metrics = display_metrics(metric_file, models, judge="avg")
      metrics = metrics[metrics["topic"] == topic]
      wandb.log({f"{topic} LR_results": wandb.Table(dataframe=new_results_df), f"{topic} metrics": wandb.Table(dataframe=metrics)})
      wandb.summary[f"{topic} acc"] = new_results_df[new_results_df["axis"] == "all_features"]["accuracy"].values[0]
      wandb.summary[f"{topic} support"] = len(test_results)
    except Exception as e:
      print(e)
      continue
  

