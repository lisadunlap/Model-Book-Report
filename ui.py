import gradio as gr
import pandas as pd
import random

# Load your dataframe
df = pd.read_csv("./data/results/harm_base-base_base-sft.csv")  # Ensure you have the correct path to your CSV
# sort df by diff_score
df = df.sort_values(by='diff_score', ascending=False)
# model_a, model_b = "Claude 2.1", "GPT-4"
# model_a_name, model_b_name = "claude-2_1", "gpt-4"
model_a, model_b = list(df.columns)[-4:-2]
model_a, model_b = model_a.replace("_counts", ""), model_b.replace("_counts", "")
model_a_name, model_b_name = model_a, model_b
    
def update_ui(cluster):
    # Filter dataframe based on selected cluster
    filtered_df = df[df['clusters'] == cluster]
    
    if not filtered_df.empty:
        # Randomly select a row (question and its details) from the filtered dataframe
        random_row = filtered_df.sample(n=1).iloc[0]
        question = random_row['question']
        model_a_output = random_row['group_1_answers']
        model_b_output = random_row['group_2_answers']
        model_a_diff = random_row['group_1_hypotheses']
        model_b_diff = random_row['group_2_hypotheses']
        model_a_count = random_row[f'{model_a_name}_counts']
        model_b_count = random_row[f'{model_b_name}_counts']

        # # remove \begin{align*} and \end{align*} from the output
        try:
            question = question.replace("$$", "$").replace("\\begin{align*}", "$\\begin{align*}").replace("\\end{align*}", "\\end{align}$")
            model_a_output = model_a_output.replace("$$", "$").replace("\\begin{align*}", "$\\begin{aligned}").replace("\\end{align*}", "\\end{aligned}$")
            model_b_output = model_b_output.replace("$$", "$").replace("\\begin{align*}", "$\\begin{aligned}").replace("\\end{align*}", "\\end{aligned}$")
        except:
            pass
        
        print("Question:", question)
        print("Model A Output:", model_a_output)
        print("Model B Output:", model_b_output)
        print("-------------------")

        return (model_a_count, model_b_count, str(question), model_a_output, model_b_output, model_a_diff, model_b_diff)
    else:
        return ("No data", "No data", "No data", "No data", "No data")

with gr.Blocks() as demo:
    gr.Markdown("# Lisa trying to convince herself that UI's matter")
    gr.Markdown("## Claude 2.1 vs GPT-4: MATH Combinatorics and Probability")
    with gr.Row():
        cluster_dropdown = gr.Dropdown(choices=df['clusters'].unique().tolist(), label="Select Cluster")
        regenerate_button = gr.Button("New Example")
    with gr.Row():
        total_count = gr.Textbox(label=f"Dataset Size", value=len(df['question'].unique()), interactive=False)
        cluster_model_a_count = gr.Textbox(label=f"{model_a} Counts", interactive=False)
        cluster_model_b_count = gr.Textbox(label=f"{model_b} Counts", interactive=False)
    with gr.Accordion('Question:', open=True):
        # add markdown whicc add a line to seoerate the question and the output
        question = gr.Markdown(label="Question", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }])
    with gr.Accordion(f'{model_a} VS {model_b} Output:', open=True):
        with gr.Row():
            model_a_output = gr.Markdown(label=f"{model_a} Output", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }], elem_id="border: 2px solid black; padding: 10px;")
            model_b_output = gr.Markdown(label=f"{model_b} Output", latex_delimiters=[{ "left": "$", "right": "$", "display": False }, { "left": "$$", "right": "$$", "display": True }], elem_id="border: 2px solid white; padding: 10px;")
    with gr.Row():
        model_a_diff = gr.Textbox(label=f"{model_a} Contains More", interactive=False)
        model_b_diff = gr.Textbox(label=f"{model_b} Contains More", interactive=False)

    # Define the callback function to update the output area when the "Regenerate" button is pressed
    regenerate_button.click(
        fn=update_ui,
        inputs=cluster_dropdown,
        outputs=[cluster_model_a_count, cluster_model_b_count, question, model_a_output, model_b_output, model_a_diff, model_b_diff],
    )

# Launch the demo
demo.launch(share=True)

