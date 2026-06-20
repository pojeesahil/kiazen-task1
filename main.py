from pathlib import Path
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
load_dotenv("secure.env")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google-genai",
    api_key=os.getenv("GOOGLE_API_KEY")
)

def get_weather(city: str)->str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
agent=create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
mainDir=Path("./data")
files=[]
for file in mainDir.iterdir():
    if file.suffix==".txt" or file.suffix==".pdf":
        files.append(str(file))
txt=[]
for path in files:
    loader=UnstructuredFileLoader(file_path=path)
    txt.extend(loader.load())
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts=text_splitter.split_documents(txt)
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore=Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("chuck saved")