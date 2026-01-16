# 🧠 Cerebrum - Memory-Augmented RAG Agent Platform

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://crebrum.vercel.app)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Cerebrum** is an advanced AI agent platform that combines episodic memory graphs with RAG-enhanced orchestration for context-aware, long-horizon reasoning. Unlike traditional RAG systems that treat memory as a stateless lookup table, Cerebrum implements persistent, temporal memory that evolves across agent sessions.

---

## 🌟 Features

### Core Capabilities
- **🧩 Episodic Memory Graphs**: Temporal recall system that maintains conversation context across sessions
- **🔗 Vector-Based Memory Retrieval**: Semantic search using LangChain for intelligent context retrieval
- **🎯 RAG-Enhanced Orchestration**: Combines retrieval-augmented generation with memory-aware agent workflows
- **⏱️ Temporal Continuity**: Tracks time-based relationships between interactions and knowledge
- **🔄 Persistent Knowledge**: Cross-session memory that builds understanding over time
- **🎨 Modern UI**: Clean, responsive interface built with React and Tailwind CSS

### Technical Architecture
- **Backend**: Python FastAPI with LangChain agent orchestration
- **Frontend**: React.js with TypeScript and modern UI components
- **Vector Database**: LanceDB for embedded vector storage and semantic search
- **Memory Storage**: SQLite for robust local data persistence
- **Deployment**: Containerized with Docker, deployed on Vercel/Render

---

## 🚀 Quick Start

### Prerequisites
Ensure you have the following installed:
- **Node.js** (v16+) and **npm**/**yarn**/**pnpm**
- **Python** (3.9+)
- **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/forUAi/crebrum.git
   cd crebrum
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Database Configuration
   DATABASE_URL=sqlite:///cerebrum.db
   
   # Vector Store Configuration
   LANCEDB_PATH=./lancedb_data
   
   # Optional: Model Configuration
   LLM_MODEL=gpt-4
   EMBEDDING_MODEL=text-embedding-3-small
   ```

### Running the Application

**Development Mode:**
```bash
# Start frontend
npm run dev

# In a separate terminal, start backend
cd backend
python main.py
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

**Production Build:**
```bash
npm run build
npm start
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  - User Interface                                        │
│  - Conversation Management                               │
│  - Memory Visualization                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ REST API
                     │
┌────────────────────▼────────────────────────────────────┐
│              Backend (FastAPI + LangChain)               │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         LangChain Agent Orchestration            │  │
│  │  - Intent Recognition                            │  │
│  │  - Tool Selection & Execution                    │  │
│  │  - Response Generation                           │  │
│  └────────────┬──────────────────┬──────────────────┘  │
│               │                  │                      │
│  ┌────────────▼─────────┐  ┌────▼──────────────────┐  │
│  │  Memory System       │  │  Vector Retrieval     │  │
│  │  - Episodic Graphs   │  │  - LanceDB            │  │
│  │  - Temporal Indexing │  │  - Semantic Search    │  │
│  │  - SQLite Storage    │  │  - Embeddings         │  │
│  └──────────────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Memory Architecture

Cerebrum implements a **multi-layered memory system** inspired by cognitive science:

1. **Working Memory** (Short-term)
   - Current conversation context
   - Active agent state
   - Immediate retrieval cache

2. **Episodic Memory** (Long-term)
   - Past interaction sequences
   - Temporal event graphs
   - Experience-based learning

3. **Semantic Memory** (Knowledge)
   - Factual information
   - Concept relationships
   - Vector embeddings for similarity

### Key Components

**LangChain Integration:**
```python
# Agent orchestration with memory
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import LanceDB

# Initialize memory-aware agent
memory = ConversationBufferMemory(return_messages=True)
vectorstore = LanceDB(embedding_function=embeddings)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    vectorstore=vectorstore
)
```

**Episodic Memory Graph:**
```python
# Temporal memory nodes with relationships
class MemoryNode:
    id: str
    content: str
    timestamp: datetime
    embeddings: List[float]
    relationships: List[str]  # Connected memory IDs
    importance_score: float
```

---

## 🔧 Configuration

### Memory Settings

Configure memory behavior in `backend/config.py`:

```python
MEMORY_CONFIG = {
    "max_short_term_messages": 10,
    "episodic_retention_days": 90,
    "similarity_threshold": 0.7,
    "max_retrieved_memories": 5,
    "consolidation_interval_hours": 24
}
```

### Model Configuration

Customize LLM and embedding models:

```python
LLM_CONFIG = {
    "model": "gpt-4",  # or "gpt-3.5-turbo", "claude-3-sonnet"
    "temperature": 0.7,
    "max_tokens": 2000
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-3-small",  # or "text-embedding-ada-002"
    "dimensions": 1536
}
```

---

## 📊 Use Cases

### Personal AI Assistant
- Remembers your preferences, habits, and past decisions
- Provides personalized recommendations based on interaction history
- Learns from feedback to improve future responses

### Research & Knowledge Management
- Builds a temporal knowledge graph of your research
- Recalls related concepts and past insights
- Connects ideas across different time periods

### Customer Support Agent
- Maintains conversation history across sessions
- Learns from past ticket resolutions
- Provides context-aware support based on user history

### Creative Writing Assistant
- Remembers character details, plot points, and world-building
- Maintains narrative consistency across chapters
- Suggests connections based on earlier story elements

---

## 🛠️ Tech Stack

### Frontend
- **React.js** - UI framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Redux Toolkit** (optional) - State management

### Backend
- **Python 3.9+** - Core language
- **FastAPI** - High-performance API framework
- **LangChain** - Agent orchestration and memory
- **LanceDB** - Vector database for embeddings
- **SQLite** - Local data persistence
- **Pydantic** - Data validation

### AI/ML
- **OpenAI GPT-4** - Language model
- **OpenAI Embeddings** - Text-to-vector conversion
- **Sentence Transformers** - Alternative embedding models

### Deployment
- **Docker** - Containerization
- **Vercel** - Frontend hosting
- **Render** - Backend API hosting

---

## 📈 Performance Optimization

### Memory Retrieval Strategies

1. **Hybrid Search**: Combines semantic similarity with temporal relevance
2. **Importance Scoring**: Prioritizes frequently accessed and recent memories
3. **Batch Retrieval**: Reduces latency by fetching multiple memories in parallel
4. **Caching**: LRU cache for frequent queries

### Scalability Considerations

- **Vector Database Sharding**: For large-scale deployments
- **Async Processing**: Non-blocking memory consolidation
- **Memory Pruning**: Automatic cleanup of low-importance memories
- **Embedding Quantization**: Reduced storage with minimal accuracy loss

---

## 🔬 Research & Inspiration

Cerebrum builds on cutting-edge research in AI agent memory:

- **Retrieval-Augmented Generation (RAG)**: Lewis et al., 2020
- **Long-Term Agentic Memory**: LangGraph/LangChain documentation
- **Continuum Memory Architectures**: Recent advancements in persistent agent memory
- **Episodic Memory Systems**: Cognitive science principles applied to AI

Key differences from traditional RAG:
- ✅ **Stateful**: Memories evolve and consolidate over time
- ✅ **Temporal**: Time-based relationships between knowledge
- ✅ **Mutable**: Retrieval strengthens memories, contradictions weaken
- ✅ **Associative**: Graph-based connections enable multi-hop reasoning

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and commit**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript/TypeScript
- Write tests for new features
- Update documentation as needed

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Vector database initialization fails**
```bash
# Solution: Ensure LanceDB directory exists
mkdir -p lancedb_data
chmod 755 lancedb_data
```

**Issue: Memory retrieval is slow**
```python
# Solution: Adjust retrieval parameters
MEMORY_CONFIG["max_retrieved_memories"] = 3
MEMORY_CONFIG["similarity_threshold"] = 0.8
```

**Issue: API connection errors**
```bash
# Solution: Verify environment variables
cat .env | grep API_KEY
# Ensure OPENAI_API_KEY is set correctly
```

---

## 📝 Roadmap

### Upcoming Features
- [ ] Multi-user support with isolated memory namespaces
- [ ] GraphRAG integration for enhanced associative recall
- [ ] Voice interface with speech-to-text
- [ ] Mobile app (React Native)
- [ ] Advanced memory visualization dashboard
- [ ] Custom tool integration framework
- [ ] Memory export/import functionality
- [ ] Fine-tuning support for domain-specific agents

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** - For the excellent agent framework and memory primitives
- **OpenAI** - For GPT-4 and embedding models
- **LanceDB** - For the high-performance vector database
- **React & Tailwind** - For modern frontend tooling

---

## 📬 Contact & Support

- **Website**: [crebrum.vercel.app](https://crebrum.vercel.app)
- **GitHub**: [github.com/forUAi/crebrum](https://github.com/forUAi/crebrum)
- **Issues**: [Report a bug or request a feature](https://github.com/forUAi/crebrum/issues)

---

## ⭐ Show Your Support

If you find Cerebrum useful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting new features
- 🤝 Contributing code

---

**Built with 🧠 by the Cerebrum team**

*Empowering AI agents with human-like memory*
