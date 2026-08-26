from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):# Check if this is a ToolMessage with artifact
            if isinstance(message.artifact, list):# The artifact should contain the list of Document objects
             context_docs.extend(message.artifact)#append
    return {
        "answer": answer,
        "context": context_docs
    }
'''You're basically saying:
"Give me the original documents that the retrieval tool found, so I can keep them as context/source information."
So in your project, you can mentally remember:
Artifact = the original retrieved documents ->dict ,imagesetc returned alongside the text result.
It's useful because the LLM can consume the serialized text, while your application can keep the original Document objects with their metadata.
Content → for the model
Artifact → for your application/code'''

if __name__ == '__main__':
    result = run_llm(query="what does EY do?")
    print(result)