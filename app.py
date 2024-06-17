from hugchat import hugchat
from hugchat.login import Login
from pyecore.resources import ResourceSet, URI
import gradio as gr
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import verify

from queue import Empty, Queue
from threading import Thread

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

dataset = load_dataset("VeryMadSoul/NLD")


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.queue.empty()

def apply_hf_settings_button(prompt, model_name) : 
    verify.chatbot.switch_llm(HF_MODELS_NAMES.index(model_name))    
    verify.chatbot.new_conversation(switch_to = True)
    return "",[]

HF_MODELS_NAMES = [model.name for model in verify.chatbot.get_available_llm_models()]


DEFAULT_TEMPERATURE = 0.2

ChatHistory = List[str]

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
# for the human, we will just inject the text
human_message_prompt_template = HumanMessagePromptTemplate.from_template("{text}")


def on_message_button_click(
    chat: Optional[ChatOpenAI],
    message: str,
    chatbot_messages: ChatHistory,
    messages: List[BaseMessage],
) -> Tuple[ChatOpenAI, str, ChatHistory, List[BaseMessage]]:
    if chat is None:
        # in the queue we will store our streamed tokens
        queue = Queue()
        # let's create our default chat
        chat = ChatOpenAI(
            model_name=GPT_MODELS_NAMES[0],
            temperature=DEFAULT_TEMPERATURE,
            streaming=True,
            callbacks=([QueueCallback(queue)]),
        )
    else:
        # hacky way to get the queue back
        queue = chat.callbacks[0].queue

    job_done = object()

    logging.info(f"Asking question to GPT, messages={messages}")
    # let's add the messages to our stuff
    messages.append(HumanMessage(content=message))
    chatbot_messages.append((message, ""))
    # this is a little wrapper we need cuz we have to add the job_done
    def task():
        chat(messages)
        queue.put(job_done)

    # now let's start a thread and run the generation inside it
    t = Thread(target=task)
    t.start()
    # this will hold the content as we generate
    content = ""
    # now, we read the next_token from queue and do what it has to be done
    while True:
        try:
            next_token = queue.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            chatbot_messages[-1] = (message, content)
            yield chat, "", chatbot_messages, messages
        except Empty:
            continue
    # finally we can add our reply to messsages
    messages.append(AIMessage(content=content))
    logging.debug(f"reply = {content}")
    logging.info(f"Done!")
    return chat, "", chatbot_messages, messages


def system_prompt_handler(value: str) -> str:
    return value


def on_clear_button_click(system_prompt: str) -> Tuple[str, List, List]:
    return "", [], [SystemMessage(content=system_prompt)]


def on_apply_settings_button_click(
    system_prompt: str, model_name: str, temperature: float
):
    logging.info(
        f"Applying settings: model_name={model_name}, temperature={temperature}"
    )
    chat = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=[QueueCallback(Queue())],
    )
    # don't forget to nuke our queue
    chat.callbacks[0].queue.empty()
    return chat, *on_clear_button_click(system_prompt)




    
def trigger_example(example):
    chat, updated_history = generate_response(example)
    return chat, updated_history

def generate_response(user_message,  history):

    #history.append((user_message,str(chatbot.chat(user_message))))
    history, errors = verify.iterative_prompting(user_message,verify.description)
    return "", history

def clear_chat():
    return [], []

examples = [dataset['train'][i]['NLD'] for i in range(len(dataset['train']))]

custom_css = """
#logo-img {
    border: none !important;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""
GPT_MODELS_NAMES = ["gpt-3.5-turbo", "gpt-4",'gpt-4o']

with gr.Blocks(analytics_enabled=False, css=custom_css) as demo:

    
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
                outputs=[chatbot],
                examples_per_page=100
            )
        #user_message.submit(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #submit_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)
        #clear_button.click(lambda x: gr.update(value=""), None, [user_message], queue=False)
        
    
    with gr.Tab("OPENAI"):
        system_prompt = gr.State(default_system_prompt)
    # here we keep our state so multiple user can use the app at the same time!
        messages = gr.State([SystemMessage(content=default_system_prompt)])
    # same thing for the chat, we want one chat per use so callbacks are unique I guess
        chat = gr.State(None)

        with gr.Column(elem_id="col_container"):
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
            
            
            chatbot1 = gr.Chatbot()
            with gr.Column():
                message = gr.Textbox(label="chat input")
                message.submit(
                    on_message_button_click,
                    [chat, message, chatbot1, messages],
                    [chat, message, chatbot1, messages],
                    queue=True,
                )
                message_button = gr.Button("Submit", variant="primary")
                message_button.click(
                    on_message_button_click,
                    [chat, message, chatbot1, messages],
                    [chat, message, chatbot1, messages],
                )
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button("Clear")
                    clear_button.click(
                        on_clear_button_click,
                        [system_prompt],
                        [message, chatbot1, messages],
                        queue=False,
                    )
                with gr.Accordion("Settings", open=False):
                    model_name = gr.Dropdown(
                        choices=GPT_MODELS_NAMES, value=GPT_MODELS_NAMES[0], label="model"
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="temperature",
                        interactive=True,
                    )
                    apply_settings_button = gr.Button("Apply")
                    apply_settings_button.click(
                        on_apply_settings_button_click,
                        [system_prompt, model_name, temperature],
                        [chat, message, chatbot1, messages],
                    )
            with gr.Row():
                gr.Examples(
                examples=examples,
                inputs=message,
                cache_examples=False,
                fn=on_message_button_click,
                outputs=[chat, message, chatbot1, messages],
                examples_per_page=100
                )
            

if __name__ == "__main__":
    # demo.launch(debug=True)
    try:
        demo.queue(api_open=False, max_size=40).launch(show_api=False)
    except Exception as e:
        print(f"Error: {e}")