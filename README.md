# Streamlit AI Assistant

A simple AI-powered assistant web app built with [Streamlit](https://streamlit.io/), [Ollama](https://ollama.com), and [LangChain](https://langchain.com). This app allows you to ask questions directly or upload a text document and get answers based on its content.

## Features

- **Chat with AI:** Ask any question and get an AI-generated answer.
- **Document Q&A:** Upload a `.txt` file and ask questions about its content.
- **Local LLM:** Uses Ollama for running language models locally.
- **Embeddings & Retrieval:** Uses HuggingFace embeddings and Chroma for document search.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed and running locally

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/streamlit-ai-assistant.git
    cd streamlit-ai-assistant
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a `.env` file in the project root (optional):

    ```
    OLLAMA_MODEL_NAME=mistral
    ```

    By default, the app uses the `mistral` model.

4. **Run the app:**
    ```bash
    streamlit run main.py
    ```

## Usage

- Enter your question in the sidebar and click **Get Answer** for a general AI response.
- Upload a `.txt` file, enter your question, and click **Ask Document** to get answers based on the uploaded document.

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Ollama](https://ollama.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Chroma](https://www.trychroma.com/)

## License

MIT License

---

Powered by [Ollama](https://ollama.com) and [LangChain](https://langchain.com).