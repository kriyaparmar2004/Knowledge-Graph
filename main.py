from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from final.GraphRag import PersonalizedGraphRAG, GraphRAGResult
# from .GraphRAG import PersonalizedGraphRAG, GraphRAGResult

 # Assuming your code is in this file

# Initialize FastAPI app
app = FastAPI(
    title="Personalized GraphRAG API",
    description="Returns personalized answers with graph and vector context.",
    version="1.0"
)

# Initialize GraphRAG system once
graphrag = PersonalizedGraphRAG(
    neo4j_url="neo4j://localhost:7687",
    neo4j_username="neo4j", 
    neo4j_password="RezinixKnowledgeGraph",
    openai_api_key="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"
)

# Request model
class QueryRequest(BaseModel):
    user_id: str
    role: str
    department: str
    experience_level: str = "intermediate"
    query: str

# Response model
class QueryResponse(BaseModel):
    answer: str
    personalization_factors: Dict
    sources: List[str]
    knowledge_graph_context: str

@app.post("/query", response_model=QueryResponse)
def query_graphrag(req: QueryRequest):
    try:
        # Ensure user profile exists or create one
        graphrag.create_user_profile(
            req.user_id, req.role, req.department, req.experience_level
        )
        
        # Run retrieval
        retrieved_chunks, graph_context = graphrag.enhanced_graph_retrieval(req.query, req.user_id)

        # Generate answer
        result: GraphRAGResult = graphrag.generate_answer(req.query, req.user_id)
        
        # Return response
        return QueryResponse(
            answer=result.answer,
            personalization_factors=result.personalization_factors,
            sources=result.sources,
            knowledge_graph_context=graph_context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/user-kg/{user_id}")
def get_user_kg(user_id: str):
    try:
        # Query graph around user and their interactions
        kg_query = f"""
        MATCH (u:User {{id: '{user_id}'}})
        OPTIONAL MATCH (u)-[:HAS_INTERACTION]->(i:Interaction)
        OPTIONAL MATCH (i)-[:CLICKED_ON]->(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        RETURN u, i, c, d, e
        LIMIT 100
        """

        results = graphrag.graph.query(kg_query)

        # Build graph data
        nodes = {}
        edges = []

        def add_node(obj, label):
            if obj and "id" in obj:
                node_id = f"{label}_{obj['id']}"
                if node_id not in nodes:
                    nodes[node_id] = {
                        "id": node_id,
                        "label": label,
                        "properties": obj
                    }

        for row in results:
            add_node(row.get('u'), "User")
            add_node(row.get('i'), "Interaction")
            add_node(row.get('c'), "Chunk")
            add_node(row.get('d'), "Document")
            add_node(row.get('e'), "Entity")

            if row.get('u') and row.get('i'):
                edges.append({"from": f"User_{row['u']['id']}", "to": f"Interaction_{row['i']['id']}", "label": "HAS_INTERACTION"})
            if row.get('i') and row.get('c'):
                edges.append({"from": f"Interaction_{row['i']['id']}", "to": f"Chunk_{row['c']['id']}", "label": "CLICKED_ON"})
            if row.get('c') and row.get('d'):
                edges.append({"from": f"Chunk_{row['c']['id']}", "to": f"Document_{row['d']['title']}", "label": "HAS_CHUNK"})
            if row.get('c') and row.get('e'):
                edges.append({"from": f"Chunk_{row['c']['id']}", "to": f"Entity_{row['e']['name']}", "label": "MENTIONS"})

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
