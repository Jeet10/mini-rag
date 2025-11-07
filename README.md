# 💬 Mini RAG — Minimal Chat

A fast and elegant **Retrieval-Augmented Generation (RAG)** chat app that lets you **chat with your documents** using real-time streamed responses — just like ChatGPT.  
Built with **Streamlit**, **FAISS**, **SentenceTransformers**, and **OpenAI / Azure OpenAI**.

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 📂 Multi-File Support | Upload `.pdf`, `.txt`, `.md` or use the built-in `docs/` folder |
| 🔍 Smart Retrieval (FAISS) | Retrieves the most relevant text chunks using vector search |
| 💬 ChatGPT-like Streaming | Responses appear **live**, token-by-token |
| 🧠 Conversation Memory | Maintains chat history and chat titles |
| 📚 Answer Modes | Switch between **Strict (RAG-only)** and **General Knowledge** modes |
| 🚫 Input Lock | Prevents overlapping queries to avoid system conflicts |
| 📝 Copy Button | One-click copy of assistant responses |
| 💾 Auto-Save Chats | Rename and reopen chats anytime |

---

## 📸 UI Preview

*(Optional — Add screenshot or GIF here later)*  
```
/screenshots
    preview.png
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mini-RAG-Chat.git
cd Mini-RAG-Chat
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key_here

# --- Optional: Azure OpenAI ---
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=your_chat_model_deployment_name
```

### 5. Add Your Documents

Place `.txt`, `.md`, and `.pdf` files inside:

```
docs/
```

Or upload them directly from the UI.

### 6. Run the app

```bash
streamlit run app.py
```

---

## 🏗 Project Structure

```
Mini-RAG-Chat/
│── app.py                     # Main Streamlit application
│── requirements.txt           # Python dependencies
│── README.md                  # Documentation
│── docs/                      # Your knowledge base files
│── chat_history/              # Auto-saved conversation logs
│── faiss.index                # Vector index (generated automatically)
│── faiss_meta.pkl             # Chunk metadata (generated automatically)
```

---

## 🔧 Technologies Used

| Component | Tool / Model |
|----------|--------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (Inner Product Search) |
| LLM Provider | OpenAI / Azure OpenAI |
| UI Framework | Streamlit |
| PDF Parsing | pypdf |

---

## 🛠 requirements.txt

```
streamlit
sentence-transformers
faiss-cpu
pypdf
python-dotenv
openai
```

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues and submit PRs.

---

## ⭐ Support the Project

If this project helped you, please **star ⭐ the repository**.  
It motivates us to keep improving!

---

## 📝 License

This project is open-source and available under the **MIT License**.

