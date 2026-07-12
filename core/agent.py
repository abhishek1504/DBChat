from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from typing import Optional

def build_agent(llm, db:SQLDatabase, verbose: bool=True):
    """Assemble the SQL agent from its parts"""
    toolkit=SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm, 
        toolkit=toolkit, 
        verbose=verbose, 
        agent_type="tool-calling"
    )

def ask(agent, question: str, callbacks: Optional[list] = None) -> str:
    """Run one question through the agent and return the answer text.
    Keeping this thin wrapper means the UI never needs to know about 
    LangChain's invoke/run API details (which have changed before and will change again e.g. when we migrate to LangGraph)"""

    result=agent.invoke({"input":question}, config={"callbacks": callbacks or []})
    return result["output"]