# LangChain Documentation Assistant

A RAG-based documentation assistant built with LangChain, OpenAI, Pinecone, Tavily, and Streamlit. It crawls EY website content, processes and chunks the documentation, generates embeddings, stores them in Pinecone, and retrieves relevant content to generate source-grounded answers using an LLM.

## .env requirement

OPENAI_API_KEY,
PINECONE_API_KEY,
TAVILY_API_KEY,
LANGSMITH_TRACING=true,
LANGSMITH_API_KEY,
LANGSMITH_PROJECT,
INDEX_NAME.

## Workflow

                 ingestion.py
                      │
                      ▼
              TavilyCrawl
                      │
                      ▼
             EY Website Content
                      │
                      ▼
          LangChain Document
                      │
                      ▼
     RecursiveCharacterTextSplitter
       chunk_size = 4000
       chunk_overlap = 200
                      │
                      ▼
           OpenAIEmbeddings
        text-embedding-3-small
                      │
                      ▼
             PineconeVectorStore
          index: langchain-doc-index
                      │
                      ▼
                 Pinecone

## When user asks Question?
                       │
                       ▼
                    main.py
                       │
                       ▼
                backend/core.py
                       │
                       ▼
              Query Embedding
                       │
                       ▼
                  Pinecone
                       │
                       ▼
             Relevant Documents
                       │
                       ▼
                  LLM (OpenAI)
                       │
                       ▼
              Generated Answer
                       │
                       ▼
                 Source URLs
                       │
                       ▼
                Streamlit UI


## Run the appication

streamlit run main.py
