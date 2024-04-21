import gradio as gr
import pandas as pd
import random
import matplotlib.pyplot as plt
import ast
import argparse

def get_summary_string(row):
    model_a_description, model_b_description, degree = "contain more", "contain more", ""

    # Determine the degree of the score
    if -0.1 < row["final_score"] < 0.1:
        degree = " slightly"
        if row["final_score"] == 0:
            model_a_description = model_b_description = "contain equal amounts of"

    # Assign descriptions based on the score's sign
    if row["final_score"] < 0:
        low_description, high_description = row["grandparent_low"], row["grandparent_high"]
    elif row["final_score"] > 0:
        low_description, high_description = row["grandparent_high"], row["grandparent_low"]
    else: # row["score"] == 0
        low_description = f"{row['grandparent_low']} and {row['grandparent_high']}"
        high_description = "" # Not used in this case

    # Construct the summary string
    print(row['grandparent_axis_name'].split('High'))
    summary_str = f"* [*{row['grandparent_axis_name'].split('High')[0].strip()}*] {model_a} {model_a_description}{degree} **{low_description}** while {model_b} {model_b_description}{degree} **{high_description}**"
    return summary_str

def get_total_summary_string(summary_clusters):
    # get markdown block of all summaries
    summary_str = "# Summary of Axes\n---\n"
    for i, row in summary_clusters.iterrows():
        summary_str += get_summary_string(row) + "\n"
    return summary_str

    
def update_ui(cluster):
    # Filter dataframe based on selected cluster
    cluster_results = results[results['grandparent_axis_name'] == cluster]
    high_description = cluster_results.iloc[0]['grandparent_high']
    low_description = cluster_results.iloc[0]['grandparent_low']
    num_questions = len(cluster_results['question'].unique())
    score = cluster_results['final_score'].mean()
    nonzero_score = cluster_results[cluster_results['final_score'] != 0]['final_score'].mean()
    # intermediate_values_cluster = intermediate_values[intermediate_values['parent_axis_name'] == row["parent_axis_name"]]
    # get question answer pairs
    questions = cluster_results.sample(2)
    question_1 = questions.iloc[0]
    question_2 = questions.iloc[1]
    response_1 = f"{question_1['grandparent_axis']}\n--------------\nScore: {question_1['final_score']}\n--------------\nFinal Score: {question_1['final_score_and_reasoning']}\n--------------\n{question_1['response']}"
    response_2 = f"{question_2['grandparent_axis']}\n--------------\nScore: {question_2['final_score']}\n--------------\nFinal Score: {question_1['final_score_and_reasoning']}\n--------------\n{question_2['response']}"

    def normalize_newlines(text):
        return text.replace("\'", "").replace("\'", "").replace('\r\n', '\n').replace('\n', '  \n')
    answer_a_1 = normalize_newlines(str(question_1['answer_a']))
    answer_b_1 = normalize_newlines(str(question_1['answer_b']))
    answer_a_2 = normalize_newlines(str(question_2['answer_a']))
    answer_b_2 = normalize_newlines(str(question_2['answer_b']))
    # remove any quotes in answers
    print(answer_b_1)
    return (question_1['question'], answer_a_1, answer_b_1, response_1, 
            question_2['question'], answer_a_2, answer_b_2, response_2,
            low_description, high_description, f"{num_questions}/{len(results['question'].unique())}", score, nonzero_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-path', type=str, default='data/results_oz.csv', help='path to data')
    parser.add_argument('--model-a-column', type=str, default='human_answers', help='column name for model A')
    parser.add_argument('--model-b-column', type=str, default='chatgpt_answers', help='column name for model B')
    args = parser.parse_args()

    # Load your dataframe
    results = pd.read_csv(args.data_path)
    results['grandparent_axis_name'] = results['grandparent_axis_name'].apply(lambda x: x.replace('**', '').replace(":", ""))
    num_questions = len(results['question'].unique())

    custom_css = """
    .input_text input {
        font-size: 4;
    }
    """

    # sort df by diff_score
    model_a, model_b = args.model_a_column, args.model_b_column
    model_a_name, model_b_name = args.model_a_column, args.model_b_column

    with gr.Blocks(theme='reilnuud/polite') as demo:
        markdown_test = """# Lisa trying to convince herself that UI's matter
        ## LLM comparrison
        Given the differenes generated for each (question, answer_A, answer_B tuple), axes are generated which represent the Axes of variation. These represent any general patterns, clusters, or variations in model outputs, with each axis having a notion of low and high.
        The 'score' of each model represents where it falls on a given axis. Start by selecting an Axis to view the scores, descriptions of the axis, and examples of questions that fall on the axis.
        """
        gr.Markdown(markdown_test, elem_id="panel")
        # gr.Markdown("# Lisa trying to convince herself that UI's matter")
        # gr.Markdown("## LLM comparrison") 
        # gr.Markdown("Given the differenes generated for each (question, answer_A, answer_B tuple), axes are generated which represent the Axes of variation. These represent any general patterns, clusters, or variations in model outputs, with each axis having a notion of low and high.")
        # gr.Markdown("The 'score' of each model represents where it falls on a given axis. Start by selecting an Axis to view the scores, descriptions of the axis, and examples of questions that fall on the axis.")
        with gr.Row():
            model_a_box = gr.Textbox(label="Model A", value=model_a, interactive=False)
            model_b_box = gr.Textbox(label="Model B", value=model_b, interactive=False)
        summary_clusters = results.groupby(['grandparent_axis_name', 'grandparent_low', 'grandparent_high']).agg({'final_score': 'mean', 'question': 'count'}).reset_index()
        gr.Markdown(get_total_summary_string(summary_clusters))
        gr.DataFrame(summary_clusters[['grandparent_axis_name', 'grandparent_low', 'grandparent_high', 'question', 'final_score']], label="Summary of Axes")
        # dis_descriptions = f"### Axis Descriptions\n---\n{logs[logs['input'] == 'Axes description'].iloc[0]['output']}"
        with gr.Row():
            cluster_dropdown = gr.Dropdown(choices=summary_clusters['grandparent_axis_name'].unique().tolist(), label="Select Axis")
            regenerate_button = gr.Button("Explore Axis")
        with gr.Row():
            description_low = gr.Textbox(label="Description Low", interactive=False)
            description_high = gr.Textbox(label="Description High", interactive=False)
        summary_str_title = gr.Markdown(f"## Summary of Axis [-1,1]\nIf the score is negative, then {model_a} is low and {model_b} is high. If the score is positive, then {model_a} is high and {model_b} is low. If the score is 0, then both models are equal.")
        with gr.Row():
            total_count = gr.Textbox(label=f"Number of Questions in the Axis", value=len(results['question'].unique()), interactive=False)
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
