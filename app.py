import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from sqlalchemy import create_engine
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with Any DB", page_icon=":robot_face:", layout="wide")
st.title("LangChain: Chat with Any DB")

POSTGRES = "USE_POSTGRES_DB"

db_uri = POSTGRES
db_host = st.sidebar.text_input("Provide DB Host")
db_user = st.sidebar.text_input("POSTGRES User")
db_password = st.sidebar.text_input("DB Password", type="password")
db_db = st.sidebar.text_input("DB Name")

api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not db_uri:
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add the groq api key")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, db_host=None, db_user=None, db_password=None, db_db=None):
    if not (db_host and db_user and db_password and db_db):
        st.error("Please provide all DB Connection details")
        st.stop()
    return SQLDatabase(create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_db}"))

db = configure_db(db_uri, db_host, db_user, db_password, db_db)

toolkit=SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"]=[{"role": "assistant", "content": "How can i help you?"}] 

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


## toolkit
