# MemoryGraph: Advanced AI Memory System

A revolutionary dual-layer memory system that combines **vector similarity search** with **knowledge graph relationships** to provide truly contextual AI responses.

## 🧠 Core Innovation

Instead of just doing semantic search on past conversations, MemoryGraph builds a **living knowledge graph** that:

- **Connects related concepts across time** using graph traversal algorithms
- **Understands relationships** between ideas, projects, and discussions
- **Provides contextual responses** by combining semantic similarity (vector search) with relational understanding (graph)

## 🏗️ Architecture

### Two-Layer Memory System

1. **Vector Memory (Short-term)**: Fast semantic retrieval using sentence transformers
2. **Graph Memory (Long-term)**: Structured knowledge with entities and relationships that persist and evolve

### Enhanced Features

- **Advanced Entity Extraction**: Uses spaCy NLP with dependency parsing for better relationship detection
- **Graph Algorithms**: Centrality measures, BFS traversal, and relationship analysis
- **Vector Similarity**: Semantic search with configurable similarity thresholds
- **Caching System**: Optimized performance with intelligent caching
- **Export/Import**: Full graph backup and restore capabilities
- **Interactive UI**: Real-time graph visualization with React Flow

## 🚀 Key Differentiator

When you ask about "data preprocessing help," MemoryGraph doesn't just find similar past messages—it:

1. **Traverses your knowledge graph** to recall you're working on earthquake prediction
2. **Retrieves your PCA notes** through vector similarity
3. **Tailors the response** to your specific context using both systems

## 📁 Project Structure

```
MemoryGraph_fron/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── memory.py            # Graph memory with advanced algorithms
│   ├── vector_memory.py     # Vector similarity search
│   ├── entity_extractor.py  # Enhanced NLP entity extraction
│   ├── llm_inference.py    # Ollama integration
│   ├── cache.py            # Performance caching
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── app/
│   │   └── page.js         # Main application
│   ├── components/
│   │   ├── ChatBox.jsx    # Chat interface
│   │   ├── MemoryGraph.jsx # Interactive graph visualization
│   │   └── ContextSidebar.jsx # Multi-tab context panel
│   ├── utils/
│   │   └── api.js         # API client functions
│   └── package.json       # Node.js dependencies
```

## 🛠️ Installation & Setup

### Backend Setup

1. **Install Python dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Install spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

3. **Install Ollama and pull a model:**
```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3.1:8b
```

4. **Start the backend (using uvicorn):**
```bash
# Option 1: Using uvicorn directly (recommended)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python main.py
```

### Frontend Setup

1. **Install Node.js dependencies:**
```bash
cd frontend
npm install
```

2. **Start the development server:**
```bash
npm run dev
```

## 🔧 API Endpoints

### Core Chat
- `POST /chat` - Main chat endpoint with dual memory retrieval

### Graph Operations
- `GET /graph/stats` - Graph statistics
- `GET /graph/view` - Full graph visualization data
- `GET /graph/central-entities` - Most central entities
- `GET /graph/related/{entity}` - Related entities
- `GET /graph/export` - Export graph data
- `POST /graph/import` - Import graph data
- `POST /graph/backup` - Create backup
- `POST /graph/restore` - Restore from backup

### Vector Search
- `POST /vector/search` - Semantic similarity search

### System
- `GET /health` - System health check
- `POST /memories/save` - Force save all memories
- `POST /cache/clear` - Clear cache

## 🎯 Usage Examples

### Example Inputs to Try

Here are example messages you can type in the chat interface to test the system:

#### **Session 1: Building Context (Start Here)**
```
I'm working on earthquake prediction using machine learning
```

```
I need help with data preprocessing for my earthquake prediction project
```

```
I'm using Python with scikit-learn and pandas for the analysis
```

#### **Session 2: Testing Memory (After Session 1)**
```
Can you help me with feature engineering for my earthquake prediction model?
```
*Notice how it remembers you're working on earthquake prediction*

```
What machine learning algorithms would work best for this project?
```

#### **Session 3: Different Domain**
```
I'm building a web application using React and Next.js
```

```
The app uses TypeScript and Tailwind CSS for styling
```

```
I need to integrate a REST API with the frontend
```

#### **Session 4: Complex Relationships**
```
I'm a data scientist working on climate change analysis using Python
```

```
My research focuses on temperature trends and carbon emissions
```

```
I use Jupyter notebooks for data visualization
```

#### **Session 5: Technical Questions**
```
How do I handle missing values in my dataset?
```

```
What's the difference between supervised and unsupervised learning?
```

#### **Example Conversation Flow:**
1. **First Message:** `"I'm building a chatbot for customer support"`
2. **Second Message:** `"It needs to handle multiple languages"`
3. **Third Message:** `"Can you help me with the NLP preprocessing?"`
   - The system will remember you're building a chatbot, it's for customer support, and needs multilingual support!

### Graph Analysis
```javascript
// Get central entities
const central = await getCentralEntities();
// Returns most important entities based on centrality measures

// Find related entities
const related = await getRelatedEntities("earthquake", 2);
// Returns entities within 2 hops of "earthquake"
```

### Vector Search
```javascript
// Search for similar content
const results = await searchVectorMemory("data preprocessing", 5);
// Returns 5 most similar past conversations with similarity scores
```

## 🧮 Advanced Features

### Graph Algorithms
- **Centrality Analysis**: Degree, betweenness, and closeness centrality
- **Graph Traversal**: BFS for finding related entities
- **Relationship Classification**: Semantic relationship types

### Vector Operations
- **Semantic Search**: Configurable similarity thresholds
- **Embedding Management**: Automatic vector generation and storage
- **Context Fusion**: Combines graph and vector context

### Performance Optimizations
- **Intelligent Caching**: Cached centrality calculations and searches
- **Background Saving**: Automatic persistence without blocking
- **Efficient Algorithms**: Optimized graph operations

## 🎨 UI Features

### Interactive Graph Visualization
- **Real-time Updates**: Graph updates as you chat
- **Node Interaction**: Click nodes to see related entities
- **Centrality Highlighting**: Important entities are visually distinct
- **Relationship Labels**: Edge labels show relationship types

### Multi-tab Context Panel
- **Context Tab**: Shows current conversation context
- **Search Tab**: Vector similarity search interface
- **Stats Tab**: Real-time graph and vector memory statistics

## 🔄 Data Flow

```
User Input → Entity Extraction → Dual Memory Update
    ↓
Vector Search + Graph Traversal → Context Fusion
    ↓
Enhanced LLM Prompt → Response + Memory Persistence
```

## 🚀 Performance

- **Caching**: Centrality calculations cached for 10 minutes
- **Background Operations**: Non-blocking saves and updates
- **Efficient Algorithms**: Optimized graph traversal
- **Memory Management**: Automatic cleanup and persistence

## 🔧 Configuration

### Environment Variables
```bash
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=60
```

### Model Configuration
- **Sentence Transformer**: `all-MiniLM-L6-v2` (configurable)
- **spaCy Model**: `en_core_web_sm`
- **Graph Persistence**: Pickle format with automatic backup

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Statistics
- Graph nodes and edges count
- Vector memory size and dimensions
- Cache hit rates and performance metrics

## 🔒 Data Persistence

- **Graph Memory**: `memory_graph.pkl`
- **Vector Memory**: `vector_memory.pkl`
- **Backups**: Timestamped JSON backups
- **Export Formats**: JSON, GEXF, GraphML

## 💡 How to Contribute

This project is experimental and open to improvement.  
If you're a developer, researcher, or AI enthusiast and want to help:

1. Fork this repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Implement your improvement
4. Open a Pull Request and describe your idea

**Suggestions for contributions:**
- Optimize graph traversal algorithms
- Improve the React graph visualization (UI/UX)
- Experiment with new embedding models or vector stores
- Add new graph analytics features (e.g., temporal evolution)
- Documentation improvements

Even small fixes or discussions are valuable — feel free to open an **Issue** with ideas or feedback.

## 📄 License

This project represents a novel approach to AI memory systems, combining the best of vector similarity search with knowledge graph relationships for truly contextual AI interactions.

---

**MemoryGraph**: Where semantic similarity meets relational understanding for the future of AI memory systems.
