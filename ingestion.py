import asyncio  #multiple tasks efficiently #multiple batchs to chroma
import os
import ssl

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates #defensive programming
#python program want to connect to a secure website
#ssl create secure tls/https connection and verify
#certifi provides a collection of trusted CA-certificate authority certificates
'''CA = the trusted organization.
*CA certificate = the thing your computer uses to recognize/trust that organization.*'''
'''ssl
 ↓
Create secure HTTPS/TLS connection
 ↓
Use certifi's trusted CA certificate bundle
 ↓
Verify website certificate
 ↓
✅ Secure connection'''
'''Mozilla CA program
       ↓
certifi CA bundle
       ↓
Python SSL
       ↓
Verify website certificate'''
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
'''Chroma = easy/local development 💻
Pinecone = managed/cloud/production scale ☁️'''
#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)#vector database local
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

'''The Document class is a standard container for a piece of text plus information about where it came from.
Document
├── page_content
│   └── "LangChain is a framework..."
│
└── metadata
    └── source: "langchain.com"
'''

async def index_documents_async(documents: list[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    '''A batch simply means a group of documents processed together.
For example, suppose your text splitter creates 1,200 document chunks.
Batch 1 → documents 1–500       (500 docs)
Batch 2 → documents 501–1000    (500 docs)
Batch 3 → documents 1001–1200   (200 docs)'''
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently(Embedding happening)
    async def add_batch(batch: list[Document], batch_num: int):
        try:
            '''Document text
     ↓
OpenAI Embeddings
     ↓
Vector
     ↓
Pinecone'''
            await vectorstore.aadd_documents(batch)#embeddings  started
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]#lets you loop over a list while getting both the index and the item.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://www.ey.com/en_in",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    '''Why "Recursive"?
It tries to split text intelligently using separators, roughly moving from larger boundaries to smaller ones when necessary.
It tries to avoid cutting text awkwardly when possible.'''
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())