import gradio as gr
import pandas as pd
import random
import matplotlib.pyplot as plt
import ast

# Load your dataframe
questions = pd.read_csv("./ui/data/all_outputs.csv")  # Ensure you have the correct path to your CSV
clusters = pd.read_csv("./ui/data/cluster_descriptions.csv")  # Ensure you have the correct path to your CSV
logs = pd.read_csv("./ui/data/llm_cluster_logs.csv")  # Ensure you have the correct path to your CSV
clusters["group_1_scores"] = clusters["group_1_scores"].apply(lambda x: ast.literal_eval(x))
clusters["group_2_scores"] = clusters["group_2_scores"].apply(lambda x: ast.literal_eval(x))
clusters["group_1_score_diffs"] = clusters["group_1_score_diffs"].apply(lambda x: ast.literal_eval(x))
clusters["group_2_score_diffs"] = clusters["group_2_score_diffs"].apply(lambda x: ast.literal_eval(x))

# sort df by diff_score
clusters = clusters.sort_values(by='difference_score', ascending=False)
model_a, model_b = "Human Answers", "ChatGPT Answers"
model_a_name, model_b_name = "Human Answers", "ChatGPT Answers"
    
def update_ui(cluster):
    # Filter dataframe based on selected cluster
    row = clusters[clusters['axis'] == cluster].iloc[0]
    description_high = row['axes_description_high']
    description_low = row['axes_description_low']
    group_1_rows = [i for i, x in enumerate(row['group_1_scores']) if (x != -100 and abs(x) > 2)]
    group_2_rows = [i for i, x in enumerate(row['group_2_scores']) if (x != -100 and abs(x) > 2)]
    selected_question_rows = set(group_1_rows + group_2_rows)
    count = len(selected_question_rows)
    group_1_avg = row['group_1_avg']
    group_2_avg = row['group_2_avg']
    
    if len(selected_question_rows) > 0:
        # Randomly select a row (question and its details) from the filtered dataframe
        sampled_question_rows = random.sample(selected_question_rows, 2)
        filtered_df = questions[questions.index.isin(sampled_question_rows)]
        random_row = filtered_df.iloc[0]
        question = random_row['question']
        model_a_output = random_row['group_1_answers']
        model_b_output = random_row['group_2_answers']
        # model_a_diff = random_row['group_1_hypotheses']
        # model_b_diff = random_row['group_2_hypotheses']
        print(row['group_1_score_diffs'])
        score_1, difference_1 = row['group_1_scores'][sampled_question_rows[0]], row['group_1_score_diffs'][sampled_question_rows[0]]
        score_2, difference_2 = row['group_2_scores'][sampled_question_rows[0]], row['group_2_score_diffs'][sampled_question_rows[0]]
        model_a_diff = f"\"{difference_1}\" \t(Score: {score_1})\n-------------------------\n{random_row['group_1_hypotheses']}" if (score_1 != -100 and abs(score_1) > 2) else f"--\n-------------------------\n{random_row['group_1_hypotheses']}"
        model_b_diff = f"\"{difference_2}\" \t(Score: {score_2})\n-------------------------\n{random_row['group_2_hypotheses']}" if (score_2 != -100 and abs(score_2) > 2) else f"--\n-------------------------\n{random_row['group_2_hypotheses']}"

        random_row_2 = filtered_df.iloc[1]
        question_2 = random_row_2['question']
        model_a_output_2 = random_row_2['group_1_answers']
        model_b_output_2 = random_row_2['group_2_answers']
        # model_a_diff_2 = random_row_2['group_1_score_diffs']
        # model_b_diff_2 = random_row_2['group_1_score_diffs']
        score_1, difference_1 = row['group_1_scores'][sampled_question_rows[1]], row['group_1_score_diffs'][sampled_question_rows[1]]
        score_2, difference_2 = row['group_2_scores'][sampled_question_rows[1]], row['group_2_score_diffs'][sampled_question_rows[1]]
        model_a_diff_2 = f"\"{difference_1}\" \t(Score: {score_1})\n-------------------------\n{random_row_2['group_1_hypotheses']}" if (score_1 != -100 and abs(score_1) > 2) else f"--\n-------------------------\n{random_row_2['group_1_hypotheses']}"
        model_b_diff_2 = f"\"{difference_2}\" \t(Score: {score_2})\n-------------------------\n{random_row_2['group_2_hypotheses']}" if (score_2 != -100 and abs(score_2) > 2) else f"--\n-------------------------\n{random_row_2['group_2_hypotheses']}"                                                                               

        # # remove \begin{align*} and \end{align*} from the output
        # try:
        #     question = question.replace("$$", "$").replace("\\begin{align*}", "$\\begin{align*}").replace("\\end{align*}", "\\end{align}$")
        #     model_a_output = model_a_output.replace("$$", "$").replace("\\begin{align*}", "$\\begin{aligned}").replace("\\end{align*}", "\\end{aligned}$")
        #     model_b_output = model_b_output.replace("$$", "$").replace("\\begin{align*}", "$\\begin{aligned}").replace("\\end{align*}", "\\end{aligned}$")
        # except:
        #     pass
        
        print("Question:", question)
        print("Model A Output:", model_a_output)
        print("Model B Output:", model_b_output)
        print("-------------------")

        plot = create_plot(row, model_a, model_b)

        print("Question:", question_2)
        return (str(question), model_a_output, model_b_output, model_a_diff, model_b_diff, 
                str(question_2), model_a_output_2, model_b_output_2, model_a_diff_2, model_b_diff_2,
                description_low, description_high, count, group_1_avg, group_2_avg, plot)
    else:
        return ("No data", "No data", "No data", "No data", "No data")
    
def create_plot(row, model_a, model_b):
    group_1_avg = row['group_1_avg']
    group_2_avg = row['group_2_avg']
    axes_description_low = row['axes_description_low']
    axes_description_high = row['axes_description_high']

    # edit the axes_description_low such that each line is at most 50 characters and words are not split
    axes_description_low = axes_description_low.split(" ")
    new_axes_description_low = []
    for i in range(0, len(axes_description_low), 5):
        new_axes_description_low.append(" ".join(axes_description_low[i:i+5]))
    axes_description_low = "\n".join(new_axes_description_low)

    # edit the axes_description_high such that each line is at most 50 characters and words are not split
    axes_description_high = axes_description_high.split(" ")
    new_axes_description_high = []
    for i in range(0, len(axes_description_high), 5):
        new_axes_description_high.append(" ".join(axes_description_high[i:i+5]))
    axes_description_high = "\n".join(new_axes_description_high)
    
    axis = row['axis']

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 4))
    # Add the data points
    ax.scatter(group_1_avg, 0, s=200, label=model_a, color='blue')
    ax.text(group_1_avg, 0.01, model_a, ha='center', va='top', fontsize=14)
    ax.scatter(group_2_avg, 0, s=200, label=model_b, color='red')
    ax.text(group_2_avg, 0.01, model_b, ha='center', va='bottom', fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8)
    # Set the descriptions at the ends of the line
    ax.text(-5, 0.01, axes_description_low, ha='right', va='bottom', fontsize=14)
    ax.text(5, 0.01, axes_description_high, ha='left', va='bottom', fontsize=14)
    # Set the limits for the x-axis
    ax.set_xlim(-5, 5)
    ax.yaxis.set_visible(False)
    # remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Add grid, legend, and title
    ax.xaxis.grid(True)
    # ax.legend(loc='lower center', ncol=2, fontsize=14)
    # turn legend off
    ax.legend().set_visible(False)
    ax.set_title(axis)
    # fix layout
    plt.tight_layout()

    return fig

with gr.Blocks(theme='reilnuud/polite') as demo:
    gr.Markdown("# Lisa trying to convince herself that UI's matter")
    gr.Markdown("## LLM comparrison") 
    gr.Markdown("Given the differenes generated for each (question, answer_A, answer_B tuple), axes are generated which represent the Axes of variation. These represent any general patterns, clusters, or variations in model outputs, with each axis having a notion of low and high.")
    gr.Markdown("The 'score' of each model represents where it falls on a given axis. Start by selecting an Axis to view the scores, descriptions of the axis, and examples of questions that fall on the axis.")
    summary_clusters = clusters[["axis", "axes_description_low", "axes_description_high", "count", "group_1_avg", "group_2_avg", "difference_score"]]
    with gr.Row():
        model_a_box = gr.Textbox(label="Model A", value=model_a, interactive=False)
        model_b_box = gr.Textbox(label="Model B", value=model_b, interactive=False)
    gr.DataFrame(summary_clusters, label="Summary of Axes")
    # dis_descriptions = f"### Axis Descriptions\n---\n{logs[logs['input'] == 'Axes description'].iloc[0]['output']}"
    with gr.Row():
        cluster_dropdown = gr.Dropdown(choices=clusters['axis'].unique().tolist(), label="Select Axis")
        regenerate_button = gr.Button("Explore Axis")
    with gr.Row():
        description_low = gr.Textbox(label="Description Low", interactive=False)
    with gr.Row():
        description_high = gr.Textbox(label="Description High", interactive=False)
    with gr.Row():
        total_count = gr.Textbox(label=f"Nymber of Questions in the Axis", value=len(questions['question'].unique()), interactive=False)
        cluster_model_a_count = gr.Textbox(label=f"{model_a} Avg Score", interactive=False)
        cluster_model_b_count = gr.Textbox(label=f"{model_b} Avg Score", interactive=False)
    with gr.Row():
        cluster_plot = gr.Plot(create_plot(clusters.iloc[0], model_a, model_b), label="Cluster Plot")
    with gr.Accordion(f'### Example 1: difference which is relevant to the cluster is listed at the top with its accompanying score', open=True):
        with gr.Accordion('Question:', open=True):
            question = gr.Markdown(label="Question", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }])
        with gr.Row():
            model_a_output = gr.Textbox(label=f"{model_a} Output", interactive=False)
            model_b_output = gr.Textbox(label=f"{model_b} Output", interactive=False)
        with gr.Row():
            model_a_diff = gr.Textbox(label=f"{model_a} Contains More", interactive=False)
            model_b_diff = gr.Textbox(label=f"{model_b} Contains More", interactive=False)
    
    with gr.Accordion(f'### Example 2: difference which is relevant to the cluster is listed at the top with its accompanying score', open=True):
        with gr.Accordion('Question:', open=True):
            question_2 = gr.Markdown(label="Question", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }])
        with gr.Row():
            model_a_output_2 = gr.Textbox(label=f"{model_a} Output", interactive=False)
            model_b_output_2 = gr.Textbox(label=f"{model_b} Output", interactive=False)
        with gr.Row():
            model_a_diff_2 = gr.Textbox(label=f"{model_a} Contains More", interactive=False)
            model_b_diff_2 = gr.Textbox(label=f"{model_b} Contains More", interactive=False)

    # Define the callback function to update the output area when the "Regenerate" button is pressed
    regenerate_button.click(
        fn=update_ui,
        inputs=cluster_dropdown,
        outputs=[question, 
                    model_a_output, 
                    model_b_output, 
                    model_a_diff,
                    model_b_diff, 
                    question_2,
                    model_a_output_2,
                    model_b_output_2,
                    model_a_diff_2,
                    model_b_diff_2,
                    description_low, 
                    description_high, 
                    total_count, 
                    cluster_model_a_count, 
                    cluster_model_b_count, 
                    cluster_plot],
    )
# Launch the demo
demo.launch(share=True)

