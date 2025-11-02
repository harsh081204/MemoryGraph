"""
FastAPI application for MemoryGraph chatbot with LLM integration.
"""
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import time
from contextlib import asynccontextmanager

from memory import MemoryGraph
from vector_memory import VectorMemory
from entity_extractor import extract_entities
from llm_inference import get_llm_response, check_ollama_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global memory instances
memory: MemoryGraph = None
vector_memory: VectorMemory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for startup and shutdown."""
    global memory, vector_memory
    
    # Startup
    logger.info("Starting MemoryGraph API...")
    memory = MemoryGraph(persist_path="memory_graph.pkl")
    vector_memory = VectorMemory(persist_path="vector_memory.pkl")
    
    # Check Ollama status
    is_running, status_msg = check_ollama_status()
    if is_running:
        logger.info(status_msg)
    else:
        logger.warning(status_msg)
    
    yield
    
    # Shutdown
    logger.info("Shutting down MemoryGraph API...")
    if memory and memory.has_unsaved_changes():
        memory.save_graph()
        logger.info("Saved memory graph on shutdown")
    if vector_memory and vector_memory.has_unsaved_changes():
        vector_memory.save_memories()
        logger.info("Saved vector memories on shutdown")


app = FastAPI(
    title="MemoryGraph API",
    description="AI chatbot with persistent knowledge graph memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js default port
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # Alternative port
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")


class ChatResponse(BaseModel):
    response: str
    context: str
    entities: list
    graph_stats: dict


# Background task to save memories periodically
def save_memories_background():
    """Background task to save both graph and vector memories."""
    if memory and memory.has_unsaved_changes():
        memory.save_graph()
    if vector_memory and vector_memory.has_unsaved_changes():
        vector_memory.save_memories()


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "name": "MemoryGraph API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "graph_stats": "/graph/stats (GET)",
            "graph_view": "/graph/view (GET)"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    ollama_running, ollama_status = check_ollama_status()
    
    return {
        "status": "healthy" if ollama_running else "degraded",
        "ollama": {
            "running": ollama_running,
            "status": ollama_status
        },
        "memory_graph": {
            "loaded": memory is not None,
            "stats": memory.get_stats() if memory else {}
        },
        "vector_memory": {
            "loaded": vector_memory is not None,
            "stats": vector_memory.get_stats() if vector_memory else {}
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest = Body(...),
    background_tasks: BackgroundTasks = None
):
    """
    Main chat endpoint.
    
    Process flow:
    1. Extract entities from user message
    2. Add entities to memory graph
    3. Retrieve relevant context from graph
    4. Generate LLM response with context
    5. Save graph in background
    """
    try:
        message = request.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Processing message: {message[:100]}...")
        
        # Step 1: Extract entities
        entities = extract_entities(message)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Step 2: Add to both memory systems
        memory.add_entities(entities)
        vector_memory.add_memory(message, {"entities": entities})
        
        # Step 3: Retrieve context from both systems
        graph_context = memory.get_context(entities, top_k=5)
        vector_context = vector_memory.get_context_from_similar(message, top_k=3)
        
        # Combine contexts
        combined_context = f"Graph Context:\n{graph_context}\n\nVector Context:\n{vector_context}"
        logger.info(f"Retrieved combined context: {len(combined_context)} chars")
        
        # Step 4: Get LLM response
        response = get_llm_response(message, combined_context)
        
        # Step 5: Schedule background save
        if background_tasks:
            background_tasks.add_task(save_memories_background)
        
        # Get graph stats
        graph_stats = memory.get_stats()
        
        return {
            "response": response,
            "context": combined_context,
            "entities": entities,
            "graph_stats": graph_stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graph/stats")
def get_graph_stats():
    """Get memory graph statistics."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    return memory.get_stats()


@app.get("/graph/view")
def view_graph():
    """View the entire memory graph (for debugging)."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    nodes = list(memory.graph.nodes())
    edges = [
        {
            "source": u,
            "target": v,
            "type": data.get("type", "RELATED_TO")
        }
        for u, v, data in memory.graph.edges(data=True)
    ]
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": memory.get_stats()
    }


@app.post("/graph/save")
def force_save_graph():
    """Manually trigger graph save."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    success = memory.save_graph()
    
    return {
        "success": success,
        "message": "Graph saved successfully" if success else "Failed to save graph"
    }


@app.get("/graph/central-entities")
def get_central_entities():
    """Get the most central entities in the graph."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    central_entities = memory.find_central_entities(top_k=10)
    
    return {
        "central_entities": [
            {"entity": entity, "centrality_score": score}
            for entity, score in central_entities
        ]
    }


@app.get("/graph/related/{entity_name}")
def get_related_entities(entity_name: str, max_distance: int = 2):
    """Get entities related to a specific entity."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    related = memory.find_related_entities(entity_name, max_distance)
    
    return {
        "entity": entity_name,
        "related_entities": related,
        "max_distance": max_distance
    }


@app.post("/vector/search")
def search_vector_memory(query: str = Body(...), top_k: int = 5):
    """Search vector memory for similar content."""
    if not vector_memory:
        raise HTTPException(status_code=503, detail="Vector memory not initialized")
    
    results = vector_memory.search_similar(query, top_k=top_k)
    
    return {
        "query": query,
        "results": results
    }


@app.post("/memories/save")
def force_save_memories():
    """Manually trigger save of both memory systems."""
    if not memory or not vector_memory:
        raise HTTPException(status_code=503, detail="Memory systems not initialized")
    
    graph_success = memory.save_graph()
    vector_success = vector_memory.save_memories()
    
    return {
        "graph_saved": graph_success,
        "vector_saved": vector_success,
        "message": "All memories saved successfully" if (graph_success and vector_success) else "Some saves failed"
    }


@app.get("/graph/export")
def export_graph(format: str = "json"):
    """Export the memory graph in various formats."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    try:
        export_data = memory.export_graph(format)
        return {
            "success": True,
            "format": format,
            "data": export_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/graph/import")
def import_graph(data: dict = Body(...)):
    """Import graph data from JSON format."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    success = memory.import_graph(data)
    
    return {
        "success": success,
        "message": "Graph imported successfully" if success else "Import failed"
    }


@app.post("/graph/backup")
def backup_graph():
    """Create a backup of the current graph."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    backup_path = f"backup_memory_graph_{int(time.time())}.json"
    success = memory.backup_graph(backup_path)
    
    return {
        "success": success,
        "backup_path": backup_path if success else None,
        "message": "Backup created successfully" if success else "Backup failed"
    }


@app.post("/graph/restore")
def restore_graph(backup_path: str = Body(...)):
    """Restore graph from backup."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    success = memory.restore_graph(backup_path)
    
    return {
        "success": success,
        "message": "Graph restored successfully" if success else "Restore failed"
    }


@app.post("/cache/clear")
def clear_cache():
    """Clear all cached data."""
    from cache import clear_cache
    clear_cache()
    
    return {
        "success": True,
        "message": "Cache cleared successfully"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)