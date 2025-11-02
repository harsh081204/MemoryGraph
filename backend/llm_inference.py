"""
LLM inference module using Ollama.
"""
import requests
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment with fallbacks
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:1b")
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))


def get_llm_response(message: str, context: str = "") -> str:
    """
    Generate LLM response using Ollama with optional knowledge graph context.
    
    Args:
        message: User's input message
        context: Optional context from knowledge graph
        
    Returns:
        Generated response string
    """
    if not message or not message.strip():
        logger.warning("Empty message received")
        return "Please provide a message."
    
    # Build prompt with context if available
    if context and context.strip() and context != "No prior context available.":
        prompt = f"""You are a helpful AI assistant with access to a knowledge graph of previous conversations.

Knowledge Graph Context:
{context}

Instructions: Use the context above when relevant to provide informed responses. If the context doesn't relate to the current question, answer from your general knowledge.

User Message: {message}

Response:"""
    else:
        prompt = f"""You are a helpful AI assistant.

User Message: {message}

Response:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        logger.info(f"Sending request to Ollama (model: {MODEL_NAME})")
        response = requests.post(
            OLLAMA_URL, 
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        llm_response = data.get("response", "").strip()
        
        if not llm_response:
            logger.warning("Received empty response from LLM")
            return "I couldn't generate a meaningful response. Please try rephrasing your question."
        
        logger.info("Successfully received LLM response")
        return llm_response
    
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {REQUEST_TIMEOUT} seconds")
        return f"The request took too long (>{REQUEST_TIMEOUT}s). The model might be loading. Please try again."
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return "Couldn't connect to Ollama. Please ensure Ollama is running (ollama serve)."
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        status_code = getattr(response, 'status_code', None)
        if status_code == 404:
            return f"Model '{MODEL_NAME}' not found. Please run: ollama pull {MODEL_NAME}"
        return f"Server error occurred (Status: {status_code}). Please try again."
    
    except Exception as e:
        logger.error(f"Unexpected error during LLM inference: {e}", exc_info=True)
        return "Sorry, I couldn't process your request right now."


def check_ollama_status() -> tuple[bool, str]:
    """
    Check if Ollama service is running and accessible.
    
    Returns:
        Tuple of (is_running, status_message)
    """
    try:
        # Try to hit the tags endpoint
        tags_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
        response = requests.get(tags_url, timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if MODEL_NAME in model_names:
            logger.info(f"Ollama is running with model '{MODEL_NAME}'")
            return True, f"Ollama is running. Model '{MODEL_NAME}' is available."
        else:
            logger.warning(f"Model '{MODEL_NAME}' not found. Available: {model_names}")
            return False, f"Ollama is running but model '{MODEL_NAME}' not found. Run: ollama pull {MODEL_NAME}"
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama")
        return False, "Cannot connect to Ollama. Please start it with: ollama serve"
    except Exception as e:
        logger.error(f"Ollama status check failed: {e}")
        return False, f"Ollama check failed: {str(e)}"