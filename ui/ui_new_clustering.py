import gradio as gr
import pandas as pd
import random
import matplotlib.pyplot as plt
import ast

# Load your dataframe
scores = pd.read_csv("./new_pipeline_intermediate_results/merged_scores.csv")
summary_clusters = pd.read_csv("./new_pipeline_intermediate_results/filtered_scores_summary.csv")
intermediate_values = pd.read_csv("./new_pipeline_intermediate_results/super_axes_new_clustering_technique_all.csv")
num_questions = len(scores['question'].unique())

custom_css = """
.input_text input {
    font-size: 4;
}
"""

# sort df by diff_score
model_a, model_b = "Human Answers", "ChatGPT Answers"
model_a_name, model_b_name = "Human Answers", "ChatGPT Answers"

def get_summary_string(row):
    model_a_description, model_b_description, degree = "contain more", "contain more", ""

    # Determine the degree of the score
    if -0.1 < row["score"] < 0.1:
        degree = " slightly"
        if row["score"] == 0:
            model_a_description = model_b_description = "contain equal amounts of"

    # Assign descriptions based on the score's sign
    if row["score"] < 0:
        low_description, high_description = row["low_description"], row["high_description"]
    elif row["score"] > 0:
        low_description, high_description = row["high_description"], row["low_description"]
    else: # row["score"] == 0
        low_description = f"{row['low_description']} and {row['high_description']}"
        high_description = "" # Not used in this case

    # Construct the summary string
    print(row['super_axis'].split('High'))
    summary_str = f"* [*{row['super_axis'].split('High')[0].strip()}*] {model_a} {model_a_description}{degree} **{low_description}** while {model_b} {model_b_description}{degree} **{high_description}**"
    return summary_str

def get_total_summary_string(summary_clusters):
    # get markdown block of all summaries
    summary_str = "# Summary of Axes\n---\n"
    for i, row in summary_clusters.iterrows():
        summary_str += get_summary_string(row) + "\n"
    return summary_str

    
def update_ui(cluster):
    # Filter dataframe based on selected cluster
    row = summary_clusters[summary_clusters['axis'] == cluster].iloc[0]
    # intermediate_values_cluster = intermediate_values[intermediate_values['super_axis'] == row["super_axis"]]
    # get question answer pairs
    questions = scores[(scores["super_axis"] == row["super_axis"]) & (scores["score"] != 0)].sample(2)
    question_1 = questions.iloc[0]
    question_2 = questions.iloc[1]
    response_1 = f"Score: {question_1['score']}\n--------------\n{question_1['response']}"
    response_2 = f"Score: {question_2['score']}\n--------------\n{question_2['response']}"

    return (question_1['question'], question_1["answer_a"], question_1["answer_b"], response_1, 
            question_2['question'], question_2["answer_a"], question_2["answer_b"], response_2,
            row["low_description"], row["high_description"], f"{row['num_questions']}/{num_questions}", row['score'], row['nonzero_score'])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Lisa trying to convince herself that UI's matter")
    gr.Markdown("## LLM comparrison") 
    gr.Markdown("Given the differenes generated for each (question, answer_A, answer_B tuple), axes are generated which represent the Axes of variation. These represent any general patterns, clusters, or variations in model outputs, with each axis having a notion of low and high.")
    gr.Markdown("The 'score' of each model represents where it falls on a given axis. Start by selecting an Axis to view the scores, descriptions of the axis, and examples of questions that fall on the axis.")
    with gr.Row():
        model_a_box = gr.Textbox(label="Model A", value=model_a, interactive=False)
        model_b_box = gr.Textbox(label="Model B", value=model_b, interactive=False)
    gr.Markdown(get_total_summary_string(summary_clusters))
    gr.DataFrame(summary_clusters[['axis', 'low_description', 'high_description', 'num_questions', 'score', 'nonzero_score']], label="Summary of Axes")
    # dis_descriptions = f"### Axis Descriptions\n---\n{logs[logs['input'] == 'Axes description'].iloc[0]['output']}"
    with gr.Row():
        cluster_dropdown = gr.Dropdown(choices=summary_clusters['axis'].unique().tolist(), label="Select Axis")
        regenerate_button = gr.Button("Explore Axis")
    with gr.Row():
        description_low = gr.Textbox(label="Description Low", interactive=False)
        description_high = gr.Textbox(label="Description High", interactive=False)
    summary_str_title = gr.Markdown(f"## Summary of Axis [-1,1]\nIf the score is negative, then {model_a} is low and {model_b} is high. If the score is positive, then {model_a} is high and {model_b} is low. If the score is 0, then both models are equal.")
    with gr.Row():
        total_count = gr.Textbox(label=f"Number of Questions in the Axis", value=len(scores['question'].unique()), interactive=False)
        cluster_score = gr.Textbox(label=f"Score", interactive=False)
        cluster_nonzero_score = gr.Textbox(label=f"Score removing neutral answers", interactive=False)
    with gr.Accordion(f'Example 1: difference which is relevant to the cluster is listed at the top with its accompanying score', open=True):
        with gr.Accordion('Question:', open=True):
            question = gr.Markdown(label="Question", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }])
        with gr.Row():
            model_a_output = gr.Textbox(label=f"{model_a} Output", interactive=False)
            model_b_output = gr.Textbox(label=f"{model_b} Output", interactive=False)
        with gr.Row():
            diff = gr.Textbox(label=f"Difference", interactive=False)
    
    with gr.Accordion(f'Example 2: difference which is relevant to the cluster is listed at the top with its accompanying score', open=True):
        with gr.Row():
            question_2 = gr.Markdown(label="Question", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }])
        with gr.Row():
            model_a_output_2 = gr.Textbox(label=f"{model_a} Output", interactive=False)
            model_b_output_2 = gr.Textbox(label=f"{model_b} Output", interactive=False)
        with gr.Row():
            diff_2 = gr.Textbox(label=f"Difference", interactive=False)

    # Define the callback function to update the output area when the "Regenerate" button is pressed
    regenerate_button.click(
        fn=update_ui,
        inputs=cluster_dropdown,
        outputs=[question, 
                    model_a_output, 
                    model_b_output, 
                    diff,
                    question_2,
                    model_a_output_2,
                    model_b_output_2,
                    diff_2,
                    description_low,
                    description_high,
                    total_count, 
                    cluster_score, 
                    cluster_nonzero_score],
    )
# Launch the demo
demo.launch(share=True)
