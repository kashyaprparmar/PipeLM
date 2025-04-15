# PipeLM 🚀

*A lightweight, modular tool for running Large Language Models (LLMs) from Hugging Face.*

PipeLM provides an intuitive CLI interface for interactive chat and a robust FastAPI server to integrate LLMs seamlessly into your applications.

---

## ✨ Overview

PipeLM simplifies interaction with AI models, allowing you to:

- 📥 **Download and manage** models from Hugging Face.
- 🌐 **Serve models** through a standardized REST API.
- 💬 **Test prompts** via an interactive chat interface.
- 📜 **Maintain conversation history**.
- 🔄 **Easily switch models** with minimal configuration changes.

---

## 🌟 Features

- 🖥️ **Interactive CLI Chat**: Engage directly from your terminal.
- 🚀 **FastAPI Server**: REST APIs with built-in health monitoring.
- 🧩 **Efficient Model Management**: Download and manage models easily.
- 📦 **Docker Support**: Containerize your models for better isolation.
- ⚡ **GPU Acceleration**: Automatically utilize available GPUs.
- 🎯 **Model Quantization**: Reduce memory usage (4-bit and 8-bit).
- 📚 **Conversation History**: Persistent chat context.
- 💡 **Rich Terminal Interface**: Enhanced CLI with markdown rendering.
- ✅ **Robust Error Handling**: Graceful handling of issues.

---

## 🛠️ Installation

### 📦 From PyPI (Recommended)
```bash
pip install pipelm
```

### 💻 From Source
```bash
git clone https://github.com/yourusername/pipelm.git
cd pipelm
pip install -e .
```

### 🐳 With Docker
```bash
git clone https://github.com/yourusername/pipelm.git
cd pipelm

docker build -f docker/Dockerfile -t pipelm .

docker run -p 8080:8080 -v pipelm_data:/root/.pipelm -e HF_TOKEN=your_token -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 pipelm
```

---

## 🚦 Usage

### 📥 Download a Model
```bash
pipelm download mistralai/Mistral-7B-Instruct-v0.2
```

### 📋 List Downloaded Models
```bash
pipelm list
```

### 💬 Interactive Chat
```bash
pipelm chat mistralai/Mistral-7B-Instruct-v0.2

# Using local model
pipelm chat /path/to/local/model

# With quantization
pipelm chat mistralai/Mistral-7B-Instruct-v0.2 --quantize 4bit
```

### 🚀 Start API Server
```bash
pipelm server mistralai/Mistral-7B-Instruct-v0.2 --port 8080

# Using local model
pipelm server /path/to/local/model --port 8080

# With quantization
pipelm server mistralai/Mistral-7B-Instruct-v0.2 --quantize 8bit
```

### 🐳 Docker Compose
```bash
export HF_TOKEN=your_token
docker-compose up -d pipelm
```

---

## 📡 API Endpoints

### 🛠️ Quick Commands

#### Check Server Health:
```bash
curl http://localhost:8080/health
```

#### Send a Sample Prompt:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain the difference between AI and machine learning."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```


### ✅ GET `/health`
Health status of server and model.

```json
{
  "status": "healthy",
  "model": "Mistral-7B-Instruct-v0.2",
  "uptime": 42.5
}
```

### 📖 GET `/`
Swagger UI for API documentation.

### ✏️ POST `/generate`
Generate text from conversation history.

Request:
```json
{
  "messages": [
    {"role": "user", "content": "What is artificial intelligence?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response:
```json
{
  "generated_text": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines..."
}
```

---

## 🎯 Chat Commands

- `/exit` or `/quit` – Exit chat
- `/clear` – Clear conversation history
- `/info` – Display current model information

---

## ⚙️ Environment Variables

- `HF_TOKEN`: Your Hugging Face token (required).
- `MODEL_DIR`: Local model directory.
- `PORT`: Server port (default: 8080).

---

## 📁 Project Structure
```
pipelm/
├── pipelm/                 # Main package
│   ├── __init__.py
│   ├── cli.py
│   ├── server.py
│   ├── downloader.py
│   ├── chat.py
│   └── utils.py
├── docker/                 # Docker setup
│   ├── Dockerfile
│   └── docker-compose.yml
├── setup.py
├── README.md
└── requirements.txt
```

---

## ✅ Requirements

- Python 3.8+
- Torch (GPU support recommended)
- 16+ GB RAM (model-dependent)
- CUDA-compatible GPU (recommended)

---

## 🚩 Troubleshooting

### Model Download Issues
- Verify Hugging Face token.
- Check network connectivity.

### Server Startup Issues
- Change default port if already in use.
- Ensure dependencies are installed.

### Memory Issues
- Use smaller models or quantization.

---

## 💽 Model Storage

- Linux/Mac: `~/.pipelm/models/`
- Windows: `%LOCALAPPDATA%\pipelm\models\`
- Docker: `/root/.pipelm/models/`

---

## 🙌 Contributing

Contributions are welcome! Submit a Pull Request.

---

## 📜 License

MIT License. See `LICENSE` for details.
