import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

# Connection string: "postgresql://<user>:<password>@<host>:5432/<db>?sslmode=require"
CONNECTION_STRING = os.getenv("AZURE_PG_CONNECTION_STRING")

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    "Get SQLite-backed chat history for a session."
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{SQLITE_DB_PATH}"
    )

def run_agent_with_memory(session_id: str, user_input: str):
    # 1. Initialize persistent memory in Cosmos DB for PostgreSQL
    # This automatically creates a 'message_store' table if it doesn't exist
    history = PostgresChatMessageHistory(
        connection_string=CONNECTION_STRING,
        session_id=session_id
    )

    # 2. Initialize the LLM (e.g., Azure OpenAI or GPT-4o)
    llm = ChatOpenAI(model="gpt-4o")

    # 3. Retrieve existing history for context
    messages = history.messages + [HumanMessage(content=user_input)]
    
    # 4. Get response from the agent
    response = llm.invoke(messages)

    # 5. Persist the interaction to the database
    history.add_user_message(user_input)
    history.add_ai_message(response.content)

    return response.content

# Example usage
if __name__ == "__main__":
    session = "user_123_thread_abc"
    reply = run_agent_with_memory(session, "Remember my name is Alice.")
    print(f"Agent: {reply}")
    
    # In a second call, the agent will remember Alice because of the DB persistence
    follow_up = run_agent_with_memory(session, "What is my name?")
    print(f"Agent: {follow_up}")
