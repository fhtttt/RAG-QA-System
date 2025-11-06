import bs4
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv

#ORM from sqlalchemy helps user to interact with database in python


load_dotenv()
api_key=os.environ.get('ZHIPU_API_KEY')

# Define the Class which can be managed by Base
Base = declarative_base() # if the class inherit from Base, the program will recognize the field 
class Session(Base):
    """
    Session Class for conversation
    """
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    """
    Message Class for the actual content in conversation
    """
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False) # system or user
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create functions for the database
def get_db():
    """
    Create a utility function to manage database sessions.  
    This function ensures that each database session is properly opened and closed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str):
    """
    Define a function to save individual messages into the database.  
    This function checks whether a session exists; if not, it creates one.  
    Then it saves the message under the corresponding session.
    """
    db = next(get_db())
    try:
        # Check if the session already exists
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            # Create a new session if it doesn't exist
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        # Add the message to the session
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Define a function to load chat history from the database.  
    This function retrieves all messages associated with the given session ID  
    and reconstructs the chat history.
    """
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        # Retrieve the session
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            # Add each message to the chat history
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Update the get_session_history function to retrieve session history from the database  
    instead of using in-memory storage only.
    """
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

def save_all_sessions():
    """
    Add functionality to save all sessions before exiting the application.  
    This function iterates through all in-memory sessions and saves their messages to the database.  
    Enhanced with error handling to ensure program stability.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import AIMessage

    print(store)
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            # Check if the message is a dictionary
            if isinstance(message, dict):
                # Verify the dictionary contains the required keys
                if "role" in message and "content" in message:
                    save_message(session_id, message["role"], message["content"])
                else:
                    print(f"Skipped a message due to missing keys: {message}")
            # Handle HumanMessage and AIMessage types
            elif isinstance(message, HumanMessage):
                save_message(session_id, "human", message.content)
            elif isinstance(message, AIMessage):
                save_message(session_id, "ai", message.content)
            else:
                print(f"Skipped an unsupported message type: {message}")

def invoke_and_save(session_id, input_text):
    """
    Modify the chained invocation function to save both the user's question  
    and the AI's response. This ensures that every interaction is recorded.
    """
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)

    # Get the AI response
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    print(f"invoke_and_save:{result}")

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)

    return result


class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            if hasattr(response, 'data') and response.data:
                embeddings.append(response.data[0].embedding)
            else:
                # Return a zero vector if embedding generation fails
                embeddings.append([0] * 1024)  # Assuming the embedding dimension is 1024
        return embeddings

    def embed_query(self, query):
        # Use the same logic, but process only a single query
        response = self.client.embeddings.create(model=self.model_name, input=query)
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding
        return [0] * 1024  # Return a zero vector if embedding generation fails







if __name__ == '__main__':

    # Step 1. Define a model instance

    chat = ChatZhipuAI(
        model_name="glm-4",
        zhipuai_api_key=api_key,  
        zhipuai_api_base="https://open.bigmodel.cn/api/paas/v4/chat/completions", 
    )



    # Step 2. Define the SQLite database and the models used to store sessions and messages.
    DATABASE_URL = "sqlite:///chat_history.db"

    # Step 3. Create ORM model classes.
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)

    # Step 4. Build a Session to manage transactions. 
    SessionLocal = sessionmaker(bind=engine)

    # Use the atexit module to register the save_all_sessions function,
    # which will run automatically when the Python program is about to exit normally.
    # The goal is to save all session data before exit to avoid data loss from sudden termination.
    import atexit

    atexit.register(save_all_sessions)

    ### Construct retriever ###
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create an embedding generator instance
    embedding_generator = EmbeddingGenerator(model_name="embedding-2")

    # List of texts
    texts = [content for document in splits for split_type, content in document if split_type == 'page_content']

    # Create a Chroma VectorStore
    chroma_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_generator,  # Use the defined embedding generator instance
        create_collection_if_not_exists=True
    )

    # Add texts to the Chroma VectorStore
    IDs = chroma_store.add_texts(texts=texts)
    # print("Added documents with IDs:", IDs)

    # Create a retriever from the Chroma VectorStore
    retriever = chroma_store.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        chat, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = invoke_and_save("abc123", "What is Task Decomposition?")
    print(result)

    # This command tells the vector store to delete its entire saved collection.
    # Here, the collection refers to all documents (text chunks) and their corresponding
    # vector representations that have been indexed and stored.
    chroma_store.delete_collection()
