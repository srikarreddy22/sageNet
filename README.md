# SageNet

An agentic RAG system for exploring philosophical texts using vector search, knowledge graphs, and LLM reasoning.

PHILOSOPHERS INCLUDED
- Aristotle
- Plato
- Buddha
- Immanuel Kant
- Confucius

SYSTEM COMPONENTS

1. Text Preprocessing (preprocess.py)
Extracts and processes PDF texts into semantic chunks. Creates 6,842 chunks from philosophical texts with metadata including philosopher name, source book, and page numbers.

2. Vector Embeddings (create_embeddings.py)
Generates embeddings using SentenceTransformers and stores them in ChromaDB for semantic search. Enables finding relevant passages based on meaning rather than keywords.

3. Knowledge Graph (build_knowledge_graph.py)
Builds a Neo4j graph database with philosophers, concepts, books, and text chunks. Contains 6,979 nodes and 58,583 relationships including WROTE, DISCUSSES, EXPLORED, PRECEDED, SHARES_CONCEPT_WITH, RELATED_TO, BELONGS_TO, and INFLUENCED.

4. Agentic RAG System (agentic_rag.py)
Combines vector search and graph queries to answer philosophical questions. Uses Gemini API as primary LLM and Groq as fallback. Classifies queries as direct, comparative, or temporal and routes to appropriate retrieval methods.

5. Web Interface (app.py)
Streamlit application providing interactive question-answering interface with conversation history and system statistics.

6. Configuration (config.py)
Manages environment variables and tests connections to Neo4j and LLM providers.

SETUP

Install dependencies:
pip install -r requirements.txt

Download spaCy model:
python -m spacy download en_core_web_sm

Create .env file with:
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

Add PDF files to philosophical_books/ folder

Run preprocessing:
cd src
python preprocess.py

Create embeddings:
python create_embeddings.py

Build knowledge graph (requires Neo4j running):
python build_knowledge_graph.py

Test system:
python config.py

Run web interface:
streamlit run app.py

USAGE

The system supports three types of queries:

Direct questions: "What is virtue according to Aristotle?"
Comparative analysis: "Compare Plato and Aristotle on justice"
Concept evolution: "How did the concept of suffering evolve over time?"

Results include LLM-generated summaries with proper citations to source texts including philosopher name, work title, and page numbers.

TECHNOLOGY STACK

Vector Search: ChromaDB with SentenceTransformers (all-MiniLM-L6-v2)
Knowledge Graph: Neo4j
LLMs: Google Gemini (primary), Groq (fallback)
NLP: spaCy for text processing
Frontend: Streamlit
Language: Python 3.12

DATA SOURCES

Philosophical texts are preprocessed from PDF format. The system extracts text, cleans artifacts, and creates semantic chunks of approximately 500 words with 100-word overlap to preserve context.

FEATURES

Semantic search across 6,842 text chunks
Graph-based relationship queries
Multi-philosopher comparison
Concept co-occurrence analysis
Temporal reasoning across philosophical eras
LLM-powered answer synthesis with citations
Interactive web interface

NOTES

Large files (PDFs, vector databases, processed data) are not included in the repository due to size constraints. Run the preprocessing pipeline to generate these locally.

API keys for Gemini and Groq are required for LLM functionality. Both offer free tiers.

Neo4j must be running locally or via cloud (AuraDB) for knowledge graph features.
