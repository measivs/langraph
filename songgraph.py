from typing import TypedDict
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

from langchain_qdrant import QdrantVectorStore
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv

load_dotenv()

user_agent={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
response = requests.get('https://genius.com/', headers=user_agent)
soup = BeautifulSoup(response.text, 'html.parser')

div = soup.find('div', class_='PageGridFull-sc-6a49b2f6-0 iWqdWA')

documents = []
text = div.get_text()
for link in div.find_all('a'):
    href = link.get('href')
    if href:
        response = requests.get(href, headers=user_agent)
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='Lyrics__Container-sc-78fb6627-1 hiRbsH')
        if div:
            lyrics_text = div.get_text()
            documents.append(Document(page_content=lyrics_text))


class State(TypedDict):
    mood: str
    answer: str
    context: str

embeddings = OpenAIEmbeddings()
client = QdrantClient(
    url="https://58b05867-8a35-4f2f-bdb7-87337af7996a.us-east4-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
)
collection_name = "songs_lyrics"
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
    context = qdrant.similarity_search(state["mood"], k=5)
    return {"context": context}

def llm_process(state: State):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant that searches songs and its lyrics based on user's mood.
        Use the following context: {context}
        Mood: {mood}
        Answer:
        """
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": state["context"], "mood": state["mood"]})
    return {"answer": response}


graph = StateGraph(State)
graph.add_node("retriever", retriever)
graph.add_node("llm_process", llm_process)
graph.add_edge(START, "retriever")
graph.add_edge("retriever", "llm_process")
graph.add_edge("llm_process", END)
graph = graph.compile()

print(graph.get_graph().draw_mermaid())

mood = "i'm sad"
result = graph.invoke({"mood": mood})
print(result)
