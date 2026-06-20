# pip install -qU langchain "langchain[google-genai]"
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
load_dotenv("secure.env")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google-genai",
    api_key=os.getenv("GOOGLE_API_KEY")
)
def get_weather(city):
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
