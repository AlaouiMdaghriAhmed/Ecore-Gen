# Meta-Model Generation Chatbot Documentation

## Overview

This Python script implements a Gradio-based web interface for a chatbot that generates meta-models from textual descriptions. The chatbot utilizes various language models, including Hugging Face models and OpenAI's GPT models, to process user inputs and generate Ecore XMI outputs.

## Key Components

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

### Main Functions

1. `generate_response` and `generate_response1`: 
   - Process user messages and generate responses using different model backends

2. `apply_hf_settings_button` and `apply_gpt_settings_button`:
   - Apply settings for Hugging Face and GPT models respectively

3. `trigger_example` and `trigger_example1`:
   - Handle example inputs for different model types

4. `clear_chat`:
   - Clear the chat history

5. File handling functions:
   - `list_files`: List files in a directory
   - `file_content`: Retrieve content of a file
   - `download_file`: Prepare a file for download

### Gradio Interface

The script creates a Gradio interface with three tabs:

1. **HF_API Tab**:
   - Interface for Hugging Face models
   - Includes chat interface, model selection, and example inputs

2. **OPENAI API Tab**:
   - Interface for OpenAI GPT models
   - Similar structure to HF_API tab

3. **File Browser Tab**:
   - Allows browsing and downloading generated files
   - Supports two directories: "HF" and "OAI"

## Usage

1. The user can select between Hugging Face and OpenAI models.
2. Input a textual description of a meta-model.
3. The chatbot processes the input and generates an Ecore XMI output.
4. Users can view and download generated files in the File Browser tab.

## Note

This script requires proper setup of environment variables and dependencies. Ensure all required libraries are installed and API keys (if needed) are properly configured before running the script.
