import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from core.agent import ask, build_agent
from core.database import DBConfig, get_database
from core.llm import get_llm

load_dotenv()

st.set_page_config(
    page_title="LangChain: Chat with Any DB",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("LangChain: Chat with Any DB")

with st.sidebar:
    st.header("Connection")
    db_config=DBConfig(
        host=st.text_input("DB Host"),
        user=st.text_input("Postgres User"),
        password=st.text_input("DB Password", type="password"),
        database=st.text_input("DB Name")
    )
    api_key=st.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY", "")
    clear_history=st.button("Clear message history")

if not api_key:
    st.info("Please add the Groq API Key.")
    st.stop()

if not db_config.is_complete():
    st.info("Please provide all DB Connection details.")
    st.stop()

@st.cache_resource(show_spinner=False)
def _get_agent(host:str, user:str, password:str, databse:str, key:str):
    # Without caching, Streamlit rebuilds the agent (and its LangGraph
    # MemorySaver) on every single rerun — i.e. on every message — which
    # would silently throw conversation memory away each time. Caching by
    # these arguments means the same agent (and checkpointer) survives
    # across reruns for the same connection, so follow-up questions
    # actually get the memory the LangGraph migration added.
    config = DBConfig(host=host, user=user, password=password, database=databse)
    db=get_database(config)
    llm=get_llm(api_key=key)
    return build_agent(llm, db)

try:
    agent=_get_agent(db_config.host, db_config.user, db_config.password, db_config.database, api_key)
except Exception as exc:
    st.error(f"Could not connect: {exc}")
    st.stop()

if "messages" not in st.session_state or clear_history:
    st.session_state.messages=[{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        callback=StreamlitCallbackHandler(st.container())
        try:
            answer=ask(agent, user_query, callbacks=[callback])
        except Exception as exc:
            answer=f"Sorry, something went wrong:{exc}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
