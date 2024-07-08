from hugchat import hugchat
from hugchat.login import Login
from pyecore.resources import ResourceSet, URI
import gradio as gr
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

import verify
import verify1
from queue import Empty, Queue
from threading import Thread
import os
import gradio as gr
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
# adapted from https://github.com/hwchase17/langchain/issues/2428#issuecomment-1512280045
from queue import Queue
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Load the NLD dataset, contains the collected use cases
dataset = load_dataset("VeryMadSoul/NLD")

# Set base directory for file operations
BASE_DIR = 'outs'

# Function to list files in a directory
def list_files(directory):
    dir_path = os.path.join(BASE_DIR, directory)
    if not os.path.exists(dir_path):
        return []
    files = os.listdir(dir_path)
    return files

# Function to read file content
def file_content(directory, file_name):
    file_path = os.path.join(BASE_DIR, directory, file_name)
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Function to get file path for download
def download_file(directory, file_name):
    file_path = os.path.join(BASE_DIR, directory, file_name)
    return file_path

# Function to apply HuggingFace model settings
def apply_hf_settings_button(prompt, model_name) : 
    verify.chatbot.switch_llm(HF_MODELS_NAMES.index(model_name))    
    verify.chatbot.new_conversation(switch_to = True)
    return "",[]

# Get available HuggingFace models
HF_MODELS_NAMES = [model.name for model in verify.chatbot.get_available_llm_models()]

# Define available GPT models
GPT_MODELS_NAMES = ["gpt-3.5-turbo", "gpt-4",'gpt-4o']

# Set default temperature for language models
DEFAULT_TEMPERATURE = 0.2

# Define type alias for chat history
ChatHistory = List[str]

# Configure logging
logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s", level=logging.INFO
)
# load up our system prompt
default_system_prompt = '''You are a systems engineer, expert in model driven engineering and meta-modeling
Your OUTPUT should always be an ecore xmi in this format :

```XML

YOUR CODE HERE 

```
'''

# Function to trigger example generation for HuggingFace models
def trigger_example1(example):
    chat, updated_history = generate_response1(example)
    return chat, updated_history

# Function to generate response using HuggingFace models
def generate_response1(user_message,  history):

    #history.append((user_message,str(chatbot.chat(user_message))))
    history, errors = verify1.iterative_prompting(user_message,verify1.description,model=verify1.model)
    return "", history

# Function to apply GPT model settings
def apply_gpt_settings_button(prompt, model_name):
    verify1.model = model_name
    return "", []

# Function to trigger example generation for GPT models
def trigger_example(example):
    chat, updated_history = generate_response(example)
    return chat, updated_history

# Function to generate response using GPT models
def generate_response(user_message,  history):

    #history.append((user_message,str(chatbot.chat(user_message))))
    history, errors = verify.iterative_prompting(user_message,verify.description)
    return "", history

# Function to clear chat history
def clear_chat():
    return [], []

# Prepare examples from the dataset
examples = [dataset['train'][i]['NLD'] for i in range(len(dataset['train']))]

# Define custom CSS for the Gradio interface
custom_css = """
#logo-img {
    border: none !important;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""


# Create Gradio interface
with gr.Blocks(analytics_enabled=False, css=custom_css) as demo:

    # HuggingFace API tab
    with gr.Tab("HF_API"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image("images\logo.png", elem_id="logo-img", show_label=False, show_share_button=False, show_download_button=False)
            with gr.Column(scale=3):
                gr.Markdown("""This Chatbot has been made to showcase our work on generating meta-model from textual descriptions.
            <br/><br/>
            The output of this conversation is going to be an ecore file that is validated by PyEcore [Pyecore (https://github.com/pyecore/pyecore)]
            <br/>
            Available Models : <br>
            - Cohere4ai-command-r-plus<br>
            - Llama-3-70B<br>
            
          """
            )
            
        with gr.Row():
            chatbot1 = gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True)
        
        with gr.Row():
            user_message = gr.Textbox(lines=1, placeholder="Ask anything ...", label="Input", show_label=False)

      
        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear chat")

            

                        
        history = gr.State([])
        
        user_message.submit(fn=generate_response, inputs=[user_message, chatbot1], outputs=[user_message, chatbot1], concurrency_limit=32)
        submit_button.click(fn=generate_response, inputs=[user_message, chatbot1], outputs=[user_message, chatbot1], concurrency_limit=32)
        
        clear_button.click(fn=clear_chat, inputs=None, outputs=[chatbot1, history], concurrency_limit=32)

        with gr.Accordion("Settings", open=False):
                            model_name = gr.Dropdown(
                                choices=HF_MODELS_NAMES, value=HF_MODELS_NAMES[0], label="model"
                            )
                            settings_button = gr.Button("Apply")
                            settings_button.click(
                                apply_hf_settings_button,
                                [user_message,model_name],
                                [user_message, chatbot1],
                            )


        
        with gr.Row():
            gr.Examples(
                examples=examples,
                inputs=user_message,
                cache_examples=False,
                fn=trigger_example,
                outputs=[chatbot1],
                examples_per_page=100
            )
        #user_message.submit(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #submit_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #clear_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)
        
    # OpenAI API tab
    with gr.Tab("OPENAI API"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image("images\logo.png", elem_id="logo-img", show_label=False, show_share_button=False, show_download_button=False)
            with gr.Column(scale=3):
                gr.Markdown("""This Chatbot has been made to showcase our work on generating meta-model from textual descriptions.
                <br/><br/>
                The output of this conversation is going to be an ecore file that is validated by PyEcore [Pyecore (https://github.com/pyecore/pyecore)]
                <br/>
                Available Models : <br>
                - GPT3-Turbo<br>
                - GPT4-Turbo<br>
                - GPT4-Omni                
                
            """
                )
            
        with gr.Row():
            chatbot1 = gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True)
        
        with gr.Row():
            user_message = gr.Textbox(lines=1, placeholder="Ask anything ...", label="Input", show_label=False)

      
        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear chat")

            

                        
        history = gr.State([])
        
        user_message.submit(fn=generate_response1, inputs=[user_message, chatbot1], outputs=[user_message, chatbot1], concurrency_limit=32)
        submit_button.click(fn=generate_response1, inputs=[user_message, chatbot1], outputs=[user_message, chatbot1], concurrency_limit=32)
        
        clear_button.click(fn=clear_chat, inputs=None, outputs=[chatbot1, history], concurrency_limit=32)
        

        
        with gr.Accordion("Settings", open=False):
                            model_name = gr.Dropdown(
                                choices=GPT_MODELS_NAMES, value=GPT_MODELS_NAMES[0], label="model"
                            )
                            settings_button = gr.Button("Apply")
                            settings_button.click(
                                apply_gpt_settings_button,
                                [user_message,model_name],
                                [user_message, chatbot1],
                            )


        with gr.Row():
            gr.Examples(
                examples=examples,
                inputs=user_message,
                cache_examples=False,
                fn=trigger_example1,
                outputs=[chatbot1],
                examples_per_page=100
            )
        #user_message.submit(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #submit_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #clear_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)

    # File Browser tab
    with gr.Tab("File Browser"):
        
        directory_dropdown = gr.Dropdown(choices=["HF", "OAI"], label="Select Directory")
        file_dropdown = gr.Dropdown(choices=[], label="Files")
        file_content_display = gr.Textbox(label="File Content", lines=10, interactive=False)
        download_button = gr.File(label="Download File")

        def update_file_list(directory):
            files = list_files(directory)
            return gr.Dropdown(choices=files)

        def update_file_content_and_path(directory, file_name):
            content = file_content(directory, file_name)
            file_path = download_file(directory, file_name)
            return content, file_path

        directory_dropdown.change(update_file_list, inputs=directory_dropdown, outputs=file_dropdown)
        file_dropdown.change(update_file_content_and_path, inputs=[directory_dropdown, file_dropdown], outputs=[file_content_display, download_button])
            
# Main execution block
if __name__ == "__main__":
    # demo.launch(debug=True)
    try:
        demo.queue(api_open=False, max_size=40).launch(show_api=False)
    except Exception as e:
        print(f"Error: {e}")
