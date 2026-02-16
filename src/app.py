"""
SageNet Streamlit Interface
Interactive frontend for philosophical exploration
"""

import streamlit as st
from agentic_rag import PhilosophicalAgent
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="SageNet - Philosophical Explorer",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_agent():
    return PhilosophicalAgent()

st.title("SageNet: Philosophical Knowledge Explorer")
st.markdown("Ask questions about philosophical texts from Aristotle, Plato, Buddha, Kant, and Confucius")

if 'agent' not in st.session_state:
    with st.spinner("Loading SageNet..."):
        st.session_state.agent = load_agent()

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")
    
    query_type = st.selectbox(
        "Query Type",
        ["Direct Question", "Comparison", "Concept Evolution"],
        help="Select the type of philosophical inquiry"
    )
    
    if query_type == "Direct Question":
        query = st.text_input(
            "Your question:",
            placeholder="What is virtue according to Aristotle?",
            key="direct_query"
        )
    elif query_type == "Comparison":
        query = st.text_input(
            "Compare philosophers:",
            placeholder="Compare Plato and Aristotle on justice",
            key="comp_query"
        )
    else:
        query = st.text_input(
            "Trace concept evolution:",
            placeholder="How did the concept of suffering evolve over time?",
            key="evo_query"
        )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("Ask SageNet", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching texts and generating answer..."):
                    answer = st.session_state.agent.answer_query(query, verbose=False)
                    st.session_state.history.append({"query": query, "answer": answer})
            else:
                st.warning("Please enter a question")
    
    with col_b:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

with col2:
    st.subheader("Example Questions")
    
    examples = [
        "What is virtue according to Aristotle?",
        "Compare Plato and Aristotle on ethics",
        "Explain Buddha's Four Noble Truths",
        "What is Kant's categorical imperative?",
        "Compare Buddhist and Stoic views on suffering",
        "How does Confucius define morality?"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}", use_container_width=True):
            st.session_state.history.append({
                "query": example,
                "answer": st.session_state.agent.answer_query(example, verbose=False)
            })
            st.rerun()

st.divider()

if st.session_state.history:
    st.subheader("Conversation History")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q: {item['query']}", expanded=(i==0)):
            st.markdown(item['answer'])
else:
    st.info("Ask a question to start exploring philosophical wisdom")

with st.sidebar:
    st.header("About SageNet")
    st.markdown("""
    SageNet is an agentic RAG system that explores philosophical texts using:
    
    - **Vector Search**: Semantic retrieval from 6,000+ text chunks
    - **Knowledge Graph**: Neo4j graph with philosophers, concepts, and relationships
    - **LLM Reasoning**: Gemini/Groq for answer synthesis
    
    **Philosophers Included:**
    - Aristotle
    - Plato
    - Buddha
    - Immanuel Kant
    - Confucius
    """)
    
    st.divider()
    
    st.subheader("System Stats")
    
    try:
        with st.spinner("Loading stats..."):
            collection = st.session_state.agent.collection
            total_chunks = collection.count()
            st.metric("Text Chunks", f"{total_chunks:,}")
            
            with st.session_state.agent.graph_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]
                st.metric("Graph Nodes", f"{node_count:,}")
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = result.single()["count"]
                st.metric("Relationships", f"{rel_count:,}")
    except Exception as e:
        st.error(f"Could not load stats: {e}")
    
    st.divider()
    
    if st.button("Refresh Agent", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()