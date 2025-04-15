# PipeLM

A lightweight API server and command-based interface for chatting with Hugging Face models.

## Overview

PipeLM provides a simple HTTP API server and interactive terminal interface for running inference with local language models. It creates a standardized way to interact with AI models, making it easy to:

- Download and manage models from Hugging Face
- Serve models through a consistent REST API
- Test prompts in an interactive chat interface
- Maintain conversation history
- Integrate language models into your applications
- Switch between different models with minimal code changes

## Features

- 🚀 FastAPI-based inference server with health monitoring
- 💬 Rich interactive terminal interface for chatting
- 🔄 Conversation history support
- 📦 Easy model download and management
- 📁 Model-specific directories to avoid conflicts
- 🧩 Support for various model backends 
- 🛠️ Easy configuration via CLI
- 📊 Basic metrics collection (tokens/sec, latency)
- ⚠️ Robust error handling

## Installation

### Method 1: Install from local directory (recommended for development)

1. Clone or create the project directory:

```bash
mkdir pipelm
cd pipelm
```

2. Create the following files:
   * `pipelm.py` (main script)
   * `app.py` (FastAPI server)
   * `setup.py` (package configuration)

3. Install the package in development mode:

```bash
pip install -e .
```

### Method 2: Run without installing

You can also run the application directly without installing it as a package:

```bash
python pipelm.py run HuggingFaceTB/SmolLM2-1.7B-Instruct
```

### Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running PipeLM

After installation, you can run PipeLM with the following command:

```bash
pipelm run HuggingFaceTB/SmolLM2-1.7B-Instruct
```

This will:
1. Download the model if not already present
2. Start the FastAPI server
3. Launch the interactive chat interface

You can also specify additional options:

```bash
pipelm run HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8081 --context-length 2048
```

### Listing Downloaded Models

To see which models you have already downloaded:

```bash
pipelm list
```

### Testing the API with the Client

To directly test the API server without using the chat interface:

```bash
pipelm client --prompt "Write a short poem about programming."
```

### Chat Commands

Within the chat interface, you can use the following commands:
* `/exit` or `/quit` - Exit the chat
* `/clear` - Clear conversation history

## API Endpoints

### GET /health
Returns health status information about the server and model.

Response:
```json
{
  "status": "healthy",
  "model": "SmolLM2-1.7B-Instruct",
  "uptime_seconds": 42.5
}
```

### POST /generate
Generates text based on a prompt.

Request body:
```json
{
  "prompt": "Write a poem about AI.",
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response:
```json
{
  "text": "Silicon dreams and neural streams...",
  "tokens_generated": 45,
  "generation_time": 1.25
}
```

## Project Structure

```
pipelm/
├── pipelm.py      # Main script with CLI commands
├── app.py         # FastAPI server implementation
├── setup.py       # Package configuration for installation
├── .gitignore
└── LICENSE
```

## Troubleshooting

### Model Download Issues

If you encounter issues downloading models:

1. Check your Hugging Face token:
   * Create or verify your token at https://huggingface.co/settings/tokens
   * Set it in your `.env` file as `HF_TOKEN=your_token_here`
2. Network issues:
   * Check your internet connection
   * Verify you have permissions to download the model

### Server Startup Issues

If the server fails to start:

1. Check if another process is using port 8080:
   * Use a different port: `pipelm run HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8081`
2. Verify Python dependencies:
   * Ensure all required packages are installed: `pip install -r requirements.txt`

### Memory Issues

If you encounter memory errors:

1. Choose a smaller model
2. Ensure you have enough RAM and GPU VRAM if using CUDA
3. Reduce the context length: `pipelm run HuggingFaceTB/SmolLM2-1.7B-Instruct --context-length 2048`

## Examples

### Basic chat:
```bash
pipelm run TheBloke/SmolLM2-1.7B-Instruct-GGUF
```

### Testing specific prompts with the client:
```bash
pipelm client --prompt "Explain quantum computing in simple terms."
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.