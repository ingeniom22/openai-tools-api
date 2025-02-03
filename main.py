import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import requests

# Load environment variables
load_dotenv()

# Configuration for Neo4j and LLM
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

neo4j_graph_store = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    enhanced_schema=True,
)


# Function to retrieve data from the knowledge graph
def retrieve_kg(question: str):
    """Retrieve data from knowledge graph to answer user question"""
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# What is the definition of contract?
MATCH (n) WHERE n.n4sch__name = "Contract" RETURN n

The question is:
{question}"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )
    try:
        chain = GraphCypherQAChain.from_llm(
            llm,
            graph=neo4j_graph_store,
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

        result = chain.invoke({"query": question})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class QueryRequest(BaseModel):
    query: str


class LKPPSearchRequest(BaseModel):
    query: str


# Endpoint to handle POST requests
@app.post("/query-kg")
async def query_knowledge_graph(request: QueryRequest):
    try:
        # Retrieve data based on user query
        result = retrieve_kg(request.query)
        return {"query": request.query, "result": result}
    except HTTPException as e:
        # Handle errors and return appropriate response
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/search-ekatalog")
async def search_ekatalog(request: LKPPSearchRequest) -> list:
    url = "https://e-katalog.lkpp.go.id/id/search-produk"

    # Query parameters
    params = {"q": request.query, "order": "relevance", "limit": "100", "offset": "0"}

    # Headers
    headers = {
        "authority": "e-katalog.lkpp.go.id",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9,id;q=0.8",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "referer": f"https://e-katalog.lkpp.go.id/id/search-produk?q={request.query}&order=relevance",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": '"Android"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
    }

    try:
        # Make the request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        soup = BeautifulSoup(response.text, "html.parser")
        card_details = soup.find_all("div", class_="card-item-detail")

        results = []

        for card in card_details:
            title = card.find("a").text
            url = card.find("a")["href"]
            desc = card.findAll("p")
            desc = "\n".join([p.text for p in desc])

            results.append({"title": title, "url": url, "desc": desc})


        # return {"query": request.query, "result": results}
        return results

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
