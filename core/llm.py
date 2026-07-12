from langchain_groq import ChatGroq

DEFAULT_MODEL="openai/gpt-oss-20b"

def get_llm(api_key:str, model_name:str=DEFAULT_MODEL, streaming:bool=True):
    """Return a LangChain chat model.
    Raises:
        ValueError: if no API key is provided"""
    if not api_key:
        raise ValueError("A groq API key is required")
    return ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=streaming)