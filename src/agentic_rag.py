"""
SageNet Agentic RAG with Gemini and Groq fallback
"""


import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


@dataclass
class RetrievalResult:
    text: str
    philosopher: str
    source: str
    page: int
    similarity: float


class LLMProvider:
    """Gemini primary, Groq fallback"""
    
    def __init__(self):
        self.gemini_available = self._init_gemini()
        self.groq_available = self._init_groq()
        
        if not self.gemini_available and not self.groq_available:
            print("Warning: No LLM available")
    
    def _init_gemini(self) -> bool:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not in .env")
            return False
        
        try:
            from google import genai
            self.gemini = genai.Client(api_key=api_key)
            self.gemini_model = "gemini-2.0-flash-exp"
            print("Gemini ready")
            return True
        except ImportError:
            print("Run: pip install google-genai")
            return False
        except Exception as e:
            print(f"Gemini error: {e}")
            return False
    
    def _init_groq(self) -> bool:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not in .env (fallback)")
            return False
        
        try:
            from groq import Groq
            self.groq = Groq(api_key=api_key)
            self.groq_model = "llama-3.1-8b-instant"
            print("Groq ready (fallback)")
            return True
        except ImportError:
            print("Run: pip install groq")
            return False
        except Exception as e:
            print(f"Groq error: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        if self.gemini_available:
            result = self._generate_gemini(prompt, max_tokens)
            if result:
                return result
            print("Gemini failed, trying Groq...")
        
        if self.groq_available:
            result = self._generate_groq(prompt, max_tokens)
            if result:
                return result
        
        return None
    
    def _generate_gemini(self, prompt: str, max_tokens: int) -> Optional[str]:
        try:
            response = self.gemini.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.warning(f"Gemini: {e}")
            return None
    
    def _generate_groq(self, prompt: str, max_tokens: int) -> Optional[str]:
        try:
            response = self.groq.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq: {e}")
            return None


class PhilosophicalAgent:
    
    def __init__(self, chroma_path: str = "../data/chroma_db"):
        print("Initializing agent...")
        
        self.llm = LLMProvider()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("philosophical_texts")
        self.graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        print("Agent ready\n")
    
    def close(self):
        self.graph_driver.close()
    
    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return "comparative"
        elif any(word in query_lower for word in ['evolution', 'change', 'over time']):
            return "temporal"
        else:
            return "direct"
    
    def vector_search(self, query: str, n_results: int = 5) -> List[RetrievalResult]:
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieved = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            retrieved.append(RetrievalResult(
                text=doc,
                philosopher=metadata['philosopher'],
                source=metadata['source_book'],
                page=metadata['page_number'],
                similarity=1 - distance
            ))
        
        return retrieved
    
    def extract_philosophers(self, query: str) -> List[str]:
        philosophers = ['Aristotle', 'Plato', 'Buddha', 'Confucius', 'Immanuel Kant', 'Kant']
        found = []
        query_lower = query.lower()
        
        for phil in philosophers:
            if phil.lower() in query_lower:
                found.append('Immanuel Kant' if phil == 'Kant' else phil)
        
        return list(set(found))
    
    def graph_query_comparative(self, philosopher1: str, philosopher2: str) -> Dict:
        with self.graph_driver.session() as session:
            result = session.run("""
                MATCH (p1:Philosopher)-[:EXPLORED]->(c:Concept)<-[:EXPLORED]-(p2:Philosopher)
                WHERE p1.name = $phil1 AND p2.name = $phil2
                RETURN collect(DISTINCT c.name) as shared_concepts
                LIMIT 1
            """, phil1=philosopher1, phil2=philosopher2)
            
            data = result.single()
            return {'shared_concepts': data['shared_concepts'] if data else []}
    
    def generate_answer(self, query: str, vector_results: List[RetrievalResult], 
                       graph_data: Dict = None) -> str:
        
        context_parts = []
        for i, result in enumerate(vector_results[:3], 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Author: {result.philosopher}\n"
                f"Work: {result.source}\n"
                f"Page: {result.page}\n"
                f"Relevance: {result.similarity:.2%}\n"
                f"Content: {result.text}\n"
            )
        
        context = "\n".join(context_parts)
        
        graph_context = ""
        if graph_data and 'shared_concepts' in graph_data and graph_data['shared_concepts']:
            concepts = graph_data['shared_concepts']
            graph_context = f"\n\nGraph Analysis: These philosophers both discuss: {', '.join(concepts[:15])}"
        
        prompt = f"""You are a philosophical scholar analyzing ancient texts. Answer this question comprehensively:

QUESTION: {query}

SOURCES FROM PHILOSOPHICAL TEXTS:
{context}
{graph_context}

YOUR TASK:
1. Write a clear summary answering the question in your own words (2-3 paragraphs)
2. Synthesize insights from multiple sources
3. Reference specific philosophers and cite properly as (Philosopher, Work, p.X)
4. Highlight key philosophical concepts
5. Compare perspectives if multiple philosophers mentioned

Format your response as:

SUMMARY:
[Your comprehensive answer in 2-3 paragraphs]

KEY POINTS:
- [Important point 1]
- [Important point 2]
- [Important point 3]

SOURCES CITED:
- [Citation 1]
- [Citation 2]
- [Citation 3]

Provide a scholarly, well-structured response."""

        answer = self.llm.generate(prompt, max_tokens=800)
        
        if not answer:
            answer = self._fallback_answer(query, vector_results, graph_data)
        
        return answer
    
    def _fallback_answer(self, query: str, vector_results: List[RetrievalResult], 
                        graph_data: Dict = None) -> str:
        
        answer = "ANSWER:\n\n"
        
        answer += f"Based on the retrieved philosophical texts, here are the most relevant passages:\n\n"
        
        for i, result in enumerate(vector_results[:3], 1):
            answer += f"[Source {i}] {result.philosopher} - {result.source} (page {result.page})\n"
            answer += f"Relevance: {result.similarity:.1%}\n"
            answer += f"{result.text}\n\n"
        
        if graph_data and 'shared_concepts' in graph_data:
            concepts = graph_data['shared_concepts']
            if concepts:
                answer += f"\nRELATED CONCEPTS:\n{', '.join(concepts[:15])}\n"
        
        answer += f"\n\nNote: LLM unavailable. Showing raw sources. "
        answer += f"Add GEMINI_API_KEY or GROQ_API_KEY to .env for synthesized answers.\n"
        
        return answer
    
    def answer_query(self, query: str, verbose: bool = True) -> str:
        if verbose:
            print(f"Query: {query}")
            
        
        query_type = self.classify_query(query)
        if verbose:
            print(f"Type: {query_type}")
        
        vector_results = self.vector_search(query, n_results=5)
        if verbose:
            print(f"Retrieved: {len(vector_results)} passages")
        
        graph_data = None
        if query_type == "comparative":
            philosophers = self.extract_philosophers(query)
            if len(philosophers) >= 2:
                if verbose:
                    print(f"Comparing: {philosophers[0]} vs {philosophers[1]}")
                graph_data = self.graph_query_comparative(philosophers[0], philosophers[1])
        
        if verbose:
            print("Generating answer...\n")
        
        answer = self.generate_answer(query, vector_results, graph_data)
        
        return answer


def main():
    agent = PhilosophicalAgent()
    
    test_queries = [
        "What is virtue according to Aristotle?",
        "Compare Plato and Aristotle on justice",
        "Explain Buddha's view on suffering"
    ]
    
    try:
        for query in test_queries:
            answer = agent.answer_query(query, verbose=True)
            print(answer)
            print("\n")
    
    finally:
        agent.close()


if __name__ == "__main__":
    main()
