# Gradio App for Ecore generation and iterative validation using LLMs
This repository contains a Gradio-based web application that leverages open-source language models from HuggingChat API and OpenAI API. The app provides three tabs: one for using models from HuggingChat and the other for models from OpenAI and the last to check to generated files and download them. Both tabs allow users to generate an Ecore file from a natural language description and iteratively validate it.

A deployed version can be found on [HF Spaces](https://huggingface.co/spaces/VeryMadSoul/Ecore-Gen)

## Setup
### Libraries and Imports

The script uses several libraries, including:
- `hugchat` for Hugging Face chat functionality
- `pyecore` for Ecore model handling
- `gradio` for creating the web interface
- `langchain` for language model interactions
- `datasets` for loading example data

### Global Variables and Settings

- `BASE_DIR`: Base directory for output files
- `HF_MODELS_NAMES`: List of available Hugging Face models
- `GPT_MODELS_NAMES`: List of available GPT models
- `DEFAULT_TEMPERATURE`: Default temperature setting for language models
- `default_system_prompt`: System prompt for the chatbot

## Features
- Two Tabs for Model Selection:
	- HuggingChat API: Uses open-source language models from HuggingChat.
	- OpenAI API: Uses language models from OpenAI.
- A Tab for checking the generated Ecore files and downloading them
- Ecore File Generation: Converts natural language descriptions into Ecore files.
- Iterative Validation: The generated Ecore files are either syntactically correct or contain an error that could not be fixed and needs human intervention.
## Installation
1. Clone the repository:

```bash
git clone https://github.com/AlaouiMdaghriAhmed/Ecore-Gen.git
cd ecore-gen
```
2.  Install the required packages:
   
It is recommeded to setup a Virtual Environment before installing the requirement : https://docs.python.org/3/tutorial/venv.html

```bash
pip install -r requirements.txt
```
## Usage

1. Navigate to the project directory:

```bash
cd ecore-gen
```
2. Run the application:

```bash
python app.py
```
3. Open your web browser and go to the provided local address to interact with the app.

## Folder Structure
- app.py: Main script to run the Gradio app.
- verify.py : Verification functions and utils for HF
- verify1.py : Verification functions and utils for OPENAI
- outs : The output files directory
  	- HF : The huggingface output dir
  	- OAI : The openai output dir
- json_dataset : The intermediate storage to be commited to HuggingFace dataset
- requirements.txt: List of required Python packages.
- README.md: Project documentation.
  
## Tool Reference
The project heavily relied on pyecore : https://github.com/pyecore/pyecore .

We also used the hugchat api : https://github.com/Soulter/hugging-chat-api .

## Gradio app on HuggingFace Spaces:
The deployed gradio app doesn't contain the huggingchat tab as the api returned some error while being setup in the spaces.

The link is : https://huggingface.co/spaces/VeryMadSoul/Ecore-Gen .

You can check the Truncated project on : https://huggingface.co/spaces/VeryMadSoul/Ecore-Gen/tree/main .

## Contributing
Contributions are welcome! Please create a new branch for each feature or bug fix:

```bash
git checkout -b feature/your-feature-name
```
Submit a pull request with a detailed explanation of your changes.



## Contact
For any questions or feedback, please open an issue or contact [mad.mik788@gmail.com].

