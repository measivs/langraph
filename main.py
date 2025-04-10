from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import os

load_dotenv()

class State(TypedDict):
    question: str
    answer: str
    context: str

embeddings = OpenAIEmbeddings()

collection_name = "my_documents"
document_paths = [
    "pdfs/hide-and-seek.txt",
    "pdfs/ginger-the-giraffe.txt",
    "pdfs/doing-my-chores.txt"
]

documents = []
for file_path in document_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        doc = Document(page_content=text)
        documents.append(doc)

client = QdrantClient(
    url="https://58b05867-8a35-4f2f-bdb7-87337af7996a.us-east4-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant = None

try:
    client.get_collection(collection_name=collection_name)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    print("Connected to existing Qdrant collection.")
except UnexpectedResponse:
    qdrant = QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url="https://58b05867-8a35-4f2f-bdb7-87337af7996a.us-east4-0.gcp.cloud.qdrant.io",
        prefer_grpc=True,
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
    )
    print("Created new Qdrant collection.")


def retriever(state: State):
    if not qdrant:
        raise ValueError("Qdrant is not initialized.")

    similar_chunks = qdrant.similarity_search(state["question"], k=2)
    context = "\n".join([chunk.page_content for chunk in similar_chunks])
    return {"context": context}

def llm_process(state: State):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = PromptTemplate.from_template(
        """
        You are a document reader, and you need to answer the user's question based on the context.
        Context: {context}
        Question: {question}
        Answer:
        """
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response}


graph_builder = StateGraph(State)
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("llm_process", llm_process)
graph_builder.add_edge(START, "retriever")
graph_builder.add_edge("retriever", "llm_process")
graph_builder.add_edge("llm_process", END)
graph = graph_builder.compile()

question = "tell me a story about hide and seek"
result = graph.invoke({"question": question})

print("Answer:", result["answer"])