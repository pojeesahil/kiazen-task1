from pathlib import Path
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
load_dotenv("secure.env")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
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
def indexDatabase():
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

    vectorstore=Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("chuck saved")
def askQuestion(question):
    vectorstore=Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})  
    pretext=(
        "You are a helpful assistant. Use the following context to answer the question.\n"
        "Context:\n{context}"
    )
    prompt=ChatPromptTemplate.from_messages([
        ("system",pretext),
        ("human","{input}"),
    ])
    qa=create_stuff_documents_chain(llm,prompt)
    ret_chain=create_retrieval_chain(retriever,qa)
    response = ret_chain.invoke({"input": question})
    print("Answer:",response["answer"])

askQuestion("which animal acts as ecosystem engineers")