from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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
)


# Function to retrieve data from the knowledge graph
def retrieve_kg(question: str):
    """Retrieve data from knowledge graph to answer user question"""
    try:
        chain = GraphCypherQAChain.from_llm(
            llm,
            graph=neo4j_graph_store,
            verbose=True,
            allow_dangerous_requests=True,
        )
        result = chain.run(question)
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
