"""
SageNet - Embedding Creation and Vector Store Setup
===================================================
This script creates embeddings from your processed text chunks
and stores them in ChromaDB for semantic search.


What this does:
1. Loads your philosophical_corpus.jsonl
2. Creates embeddings using SentenceTransformers
3. Stores in ChromaDB (vector database)
4. Enables semantic search capabilities
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhilosophicalEmbeddingCreator:
  

    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chroma_persist_dir: str = "../data/chroma_db"):
        """
        Initialize the embedding creator.
        
        Args:
            model_name: SentenceTransformer model to use
                       (all-MiniLM-L6-v2 is a great balance of speed/quality)
            chroma_persist_dir: Where to save the ChromaDB database
        """
        self.model_name = model_name
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        logger.info(f"Initializing embedding creator with model: {model_name}")
        
        # Load the embedding model
        # This downloads the model the first time (about 90MB)
        logger.info("Loading SentenceTransformer model (first time will download ~90MB)...")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ Model loaded successfully")
        
        # Initialize ChromaDB client
        logger.info(f"Setting up ChromaDB at: {self.chroma_persist_dir.absolute()}")
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_persist_dir))
        
        # Create or get collection
        # A collection is like a table in a database
        self.collection_name = "philosophical_texts"
        
    def load_corpus(self, corpus_path: str) -> List[Dict]:
        """
        Load the processed corpus from JSONL file.
        
        Args:
            corpus_path: Path to philosophical_corpus.jsonl
        
        Returns:
            List of chunk dictionaries
        """
        corpus_path = Path(corpus_path)
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        logger.info(f"Loading corpus from: {corpus_path}")
        chunks = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
        
        logger.info(f"✅ Loaded {len(chunks):,} chunks")
        return chunks
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Create embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
                       (32 is good for most laptops)
        
        Returns:
            List of embedding vectors
        """
        logger.info(f"Creating embeddings for {len(texts):,} texts...")
        
        all_embeddings = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            
            # Create embeddings for this batch
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,  # We have our own progress bar
                convert_to_numpy=True
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
        
        logger.info(f"✅ Created {len(all_embeddings):,} embeddings")
        return all_embeddings
    
    def create_vector_store(self, chunks: List[Dict], batch_size: int = 100):
        """
        Create embeddings and store in ChromaDB.
        
        Args:
            chunks: List of text chunks from corpus
            batch_size: How many chunks to process at once
        
        This is the main function that does everything!
        """
        logger.info("="*70)
        logger.info("CREATING VECTOR STORE")
        logger.info("="*70)
        
        # Delete existing collection if it exists (fresh start)
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection for fresh start")
        except:
            pass
        
        # Create new collection
        logger.info(f"Creating collection: {self.collection_name}")
        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Philosophical texts from multiple philosophers"}
        )
        
        # Prepare data
        logger.info("Preparing data for embedding...")
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Prepare metadata (ChromaDB needs flat dictionaries)
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                'philosopher': chunk['philosopher'],
                'source_book': chunk['source_book'],
                'page_number': chunk['page_number'],
                'chunk_index': chunk['chunk_index'],
                'word_count': chunk['word_count']
            })
        
        # Create embeddings in batches and add to ChromaDB
        logger.info(f"Processing {len(chunks):,} chunks in batches of {batch_size}...")
        
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(chunks), batch_size), 
                     desc="Creating and storing embeddings",
                     total=total_batches):
            
            batch_end = min(i + batch_size, len(chunks))
            
            # Get batch data
            batch_texts = texts[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            # Create embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to ChromaDB
            collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        logger.info("✅ Vector store created successfully!")
        logger.info(f"📊 Total documents in collection: {collection.count()}")
        
        return collection
    
    def test_semantic_search(self, collection, test_queries: List[str], n_results: int = 3):
        """
        Test the vector store with sample queries.
        
        Args:
            collection: ChromaDB collection
            test_queries: List of test questions
            n_results: Number of results to return per query
        
        This shows you that semantic search is working!
        """
        logger.info("\n" + "="*70)
        logger.info("TESTING SEMANTIC SEARCH")
        logger.info("="*70)
        
        for query in test_queries:
            logger.info(f"\n🔍 Query: '{query}'")
            logger.info("-"*70)
            
            # Create embedding for the query
            query_embedding = self.model.encode(query).tolist()
            
            # Search the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Display results
            for idx, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                logger.info(f"\nResult {idx}:")
                logger.info(f"  Philosopher: {metadata['philosopher']}")
                logger.info(f"  Source: {metadata['source_book']}")
                logger.info(f"  Page: {metadata['page_number']}")
                logger.info(f"  Similarity: {1 - distance:.3f}")  # Convert distance to similarity
                logger.info(f"  Text: {doc[:200]}...")
                logger.info("-"*70)
    
    def get_statistics(self, collection):
        """
        Print statistics about the vector store.
        """
        logger.info("\n" + "="*70)
        logger.info("VECTOR STORE STATISTICS")
        logger.info("="*70)
        
        total_docs = collection.count()
        logger.info(f"Total documents: {total_docs:,}")
        logger.info(f"Embedding model: {self.model_name}")
        logger.info(f"Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        logger.info(f"Storage location: {self.chroma_persist_dir.absolute()}")
        
        # Get a sample to show metadata
        sample = collection.get(limit=1)
        if sample['metadatas']:
            logger.info(f"\nSample metadata fields:")
            for key in sample['metadatas'][0].keys():
                logger.info(f"  - {key}")
        
        logger.info("="*70)


def main():
    """
    Main execution function - runs the entire pipeline.
    """
    # ==================== CONFIGURATION ====================
    CORPUS_PATH = "../data/philosophical_corpus.jsonl"
    CHROMA_DIR = "../data/chroma_db"
    BATCH_SIZE = 100  # Adjust based on your RAM (100 is safe for most machines)
    
    # Test queries to verify semantic search works
    TEST_QUERIES = [
        "What is virtue according to Aristotle?",
        "Buddha's views on suffering and enlightenment",
        "How does Plato define justice?",
        "Kant's categorical imperative explained",
        "Confucius on moral cultivation"
    ]
    # =======================================================
    
    logger.info("="*70)
    logger.info("SageNet Embedding Creation Pipeline")
    logger.info("="*70)
    logger.info(f"Corpus: {Path(CORPUS_PATH).absolute()}")
    logger.info(f"Vector DB: {Path(CHROMA_DIR).absolute()}")
    logger.info("="*70 + "\n")
    
    # Initialize the embedding creator
    creator = PhilosophicalEmbeddingCreator(
        model_name="all-MiniLM-L6-v2",
        chroma_persist_dir=CHROMA_DIR
    )
    
    # Step 1: Load corpus
    chunks = creator.load_corpus(CORPUS_PATH)
    
    # Step 2: Create vector store
    start_time = time.time()
    collection = creator.create_vector_store(chunks, batch_size=BATCH_SIZE)
    elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Total time: {elapsed/60:.2f} minutes")
    
    # Step 3: Get statistics
    creator.get_statistics(collection)
    
    # Step 4: Test with sample queries
    creator.test_semantic_search(collection, TEST_QUERIES, n_results=2)
    
    logger.info("\n" + "="*70)
    logger.info("✅ EMBEDDING CREATION COMPLETE!")
    logger.info("="*70)
    logger.info("Your vector store is ready for the RAG system.")
    logger.info(f"Location: {Path(CHROMA_DIR).absolute()}")
    logger.info("\nNext step: Build the agentic RAG pipeline!")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
