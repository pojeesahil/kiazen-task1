# Local RAG Chatbot with Gemini & LangChain

This project is a Retrieval-Augmented Generation (RAG) chat system that allows you to ask questions about your local documents (PDFs and Text files). It uses Google's Gemini API for embeddings and text generation, LangChain for the pipeline, and ChromaDB for local vector storage.

**Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/))


## Setup Instructions

### Download
Download this Project as zip or

fetch from github by
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd kaizen-task1
```

### install all requirements
pip install -r requirements.txt


### storing secret key
make a file secure.env on main folder and write GOOGLE_API_KEY=[your_key] inside it


### Running
now you can execute the prograam by

```python main.py```

You can type **index** for reIndexing of database or type **exit** for exit or you can just normally ask any question

### Terminal output
<img width="1864" height="578" alt="Screenshot 2026-06-22 212644" src="https://github.com/user-attachments/assets/5d937d9e-f4f4-4de7-aaec-1db2fd6748d2" />



