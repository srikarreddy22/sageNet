"""
This script extracts, cleans, and chunks philosophical texts from PDFs.

"""


import fitz  
import re
import json
import spacy
from pathlib import Path
from typing import List, Dict
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging - shows you what's happening in real-time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Represents a single chunk of philosophical text.
    Think of this as a "smart paragraph" with context.
    
    Fields:
        chunk_id: Unique identifier (e.g., "aristotle_42")
        philosopher: Name of the philosopher
        source_book: Name of the book file
        text: The actual text content
        page_number: Page in original PDF
        chunk_index: Sequential number of this chunk
        word_count: Number of words in chunk
        metadata: Additional information (dict)
    """
    chunk_id: str
    philosopher: str
    source_book: str
    text: str
    page_number: int
    chunk_index: int
    word_count: int
    metadata: Dict


class PhilosophicalTextPreprocessor:
    """
    Main preprocessing engine for philosophical texts.
    
    This class handles the entire pipeline:
    1. PDF Extraction - Gets text from PDF files
    2. Text Cleaning - Removes artifacts, fixes formatting
    3. Semantic Chunking - Splits into meaningful pieces
    4. Metadata Addition - Adds context information
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the preprocessor.
        
        Args:
            chunk_size: Target number of words per chunk
                       (500 is a good balance for philosophical texts)
            chunk_overlap: Number of words to overlap between chunks
                          (100 words helps maintain context continuity)
        
        Example:
            preprocessor = PhilosophicalTextPreprocessor(
                chunk_size=500,
                chunk_overlap=100
            )
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load spaCy for intelligent sentence segmentation
        # spaCy understands sentence boundaries better than simple split('.')
        logger.info("Loading spaCy language model...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            # We disable NER and lemmatizer because we don't need them = faster
        except OSError:
            logger.error("spaCy model not found! Run: python -m spacy download en_core_web_sm")
            raise
        
        # Philosopher name mapping (add more as needed)
        self.philosopher_mapping = {
            "aristotle": "Aristotle",
            "buddha": "Buddha",
            "plato": "Plato",
            "kant": "Immanuel Kant",
            "confucius": "Confucius"
        }
    
    def identify_philosopher(self, filename: str) -> str:
        
        filename_lower = filename.lower()
        for key, name in self.philosopher_mapping.items():
            if key in filename_lower:
                return name
        return "Unknown"
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF page by page.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            List of dicts: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        
    
        """
        logger.info(f"Extracting text from: {pdf_path}")
        pages_data = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc, start=1):
                # Get text from current page
                text = page.get_text()
                
                # Only add pages that have actual content
                if text.strip():
                    pages_data.append({
                        "page": page_num,
                        "text": text
                    })
                
                # Log progress every 50 pages for large books
                if page_num % 50 == 0:
                    logger.info(f"  Progress: {page_num}/{total_pages} pages")
            
            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages with content")
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            raise
        
        return pages_data
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving philosophical content.
        
        Args:
            text: Raw text from PDF
        
        Returns:
            Cleaned text ready for chunking
        
        
        """
        # Remove page number patterns like "Page 42" or "- 42 -"
        text = re.sub(r'(?i)page\s*\d+', '', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines (one blank line)
        text = re.sub(r' {2,}', ' ', text)      # Max 1 space between words
        
        # Remove common header/footer artifacts
        text = re.sub(r'(?i)(chapter|section)\s*\d+\s*\n', '', text)
        
        # Fix hyphenated line breaks (very common in PDFs)
        # Example: "philoso-\nphy" -> "philosophy"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove standalone numbers (often page number artifacts)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove URLs (sometimes in footnotes)
        text = re.sub(r'https?://\S+', '', text)
        
        # Clean up but preserve structure
        text = text.strip()
        
        return text
    
    def create_semantic_chunks(self, text: str, page_num: int) -> List[Dict]:
        """
        Create semantically coherent chunks using sentence boundaries.
        
        Args:
            text: Cleaned text to chunk
            page_num: Page number (for metadata)
        
        Returns:
            List of chunk dictionaries
        
        
        """
        # Process text with spaCy to get sentence boundaries
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "word_count": current_word_count
                })
                
                # Implement overlap: keep last few sentences for context
                overlap_words = 0
                overlap_sentences = []
                
                # Go backwards through sentences to build overlap
                for sent in reversed(current_chunk):
                    overlap_words += len(sent.split())
                    overlap_sentences.insert(0, sent)
                    if overlap_words >= self.chunk_overlap:
                        break
                
                # Start next chunk with overlap
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Don't forget the last chunk!
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "word_count": current_word_count
            })
        
        return chunks
    
    def process_book(self, pdf_path: str) -> List[TextChunk]:
        """
        Complete processing pipeline for one book.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of TextChunk objects
        """
        pdf_path = Path(pdf_path)
        philosopher = self.identify_philosopher(pdf_path.name)
        logger.info(f"Processing book by {philosopher}: {pdf_path.name}")
        
        # Stage 1: Extract text from PDF
        pages_data = self.extract_text_from_pdf(str(pdf_path))
        
        if not pages_data:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []
        
        all_chunks = []
        chunk_counter = 0
        
        # Process each page
        for page_data in pages_data:
            page_num = page_data["page"]
            raw_text = page_data["text"]
            
            # Stage 2: Clean the text
            cleaned_text = self.clean_text(raw_text)
            
            # Skip pages with too little content (likely TOC, blank pages, etc.)
            if not cleaned_text or len(cleaned_text.split()) < 20:
                continue
            
            # Stage 3: Create semantic chunks
            page_chunks = self.create_semantic_chunks(cleaned_text, page_num)
            
            # Stage 4: Add metadata and create TextChunk objects
            for chunk_data in page_chunks:
                chunk = TextChunk(
                    chunk_id=f"{philosopher.lower().replace(' ', '_')}_{chunk_counter}",
                    philosopher=philosopher,
                    source_book=pdf_path.stem,  # Filename without .pdf extension
                    text=chunk_data["text"],
                    page_number=chunk_data["page"],
                    chunk_index=chunk_counter,
                    word_count=chunk_data["word_count"],
                    metadata={
                        "original_file": pdf_path.name,
                        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                        "chunk_method": "semantic_sentence_based",
                        "chunk_size_target": self.chunk_size,
                        "overlap_size": self.chunk_overlap
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {philosopher}'s text")
        return all_chunks
    
    def save_to_jsonl(self, chunks: List[TextChunk], output_path: str):
        """
        Save chunks to JSONL format.
        
        Args:
            chunks: List of TextChunk objects
            output_path: Where to save the file
        
        """
        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        
        # Create parent directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Convert dataclass to dict and write as JSON line
                json_line = json.dumps(asdict(chunk), ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Successfully saved to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def process_all_books(self, pdf_directory: str, output_file: str = "philosophical_corpus.jsonl"):
        """
        Process all PDF books in a directory.
        
        Args:
            pdf_directory: Path to folder containing PDFs
            output_file: Where to save the final JSONL file
        
        This is your main entry point!
        Just point it at a folder of PDFs and it does everything.
        """
        pdf_dir = Path(pdf_directory)
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_directory}")
            logger.error(f"Make sure your PDFs are in: {pdf_dir.absolute()}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.name}")
        
        all_chunks = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            try:
                book_chunks = self.process_book(str(pdf_file))
                all_chunks.extend(book_chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                logger.error("Continuing with other files...")
                continue
        
        if not all_chunks:
            logger.error("No chunks were created! Check your PDF files.")
            return
        
        # Save all chunks to single JSONL file
        self.save_to_jsonl(all_chunks, output_file)
        
        # Print summary statistics
        self.print_statistics(all_chunks)
    
    def print_statistics(self, chunks: List[TextChunk]):
        """
        Print helpful statistics about the processed corpus.
        
        Args:
            chunks: List of all processed chunks
        

        """
        total_chunks = len(chunks)
        total_words = sum(chunk.word_count for chunk in chunks)
        
        # Count chunks per philosopher
        philosopher_counts = {}
        for chunk in chunks:
            philosopher_counts[chunk.philosopher] = philosopher_counts.get(chunk.philosopher, 0) + 1
        
        # Print beautiful statistics
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE - CORPUS STATISTICS")
        logger.info("="*60)
        logger.info(f"Total chunks created: {total_chunks:,}")
        logger.info(f"Total words processed: {total_words:,}")
        logger.info(f"Average words per chunk: {total_words//total_chunks if total_chunks > 0 else 0}")
        logger.info(f"\nChunks per philosopher:")
        
        for philosopher, count in sorted(philosopher_counts.items()):
            percentage = (count / total_chunks) * 100
            logger.info(f"  {philosopher:20s}: {count:5,} chunks ({percentage:5.1f}%)")
        
        logger.info("="*60 + "\n")
        logger.info("Next steps:")
        logger.info("1. Check the output file to verify quality")
        logger.info("2. Ready to create embeddings and build the RAG system!")
        logger.info("="*60 + "\n")


# ==============================================================================
# MAIN EXECUTION - Run this script!
# ==============================================================================

if __name__ == "__main__":
    
    PDF_DIRECTORY = "../philosophical_books"  # Where your PDFs are
    OUTPUT_FILE = "../data/philosophical_corpus.jsonl"  # Where to save results
    
    # Chunking parameters (you can experiment with these)
    CHUNK_SIZE = 500      # Target words per chunk
    CHUNK_OVERLAP = 100   # Words to overlap between chunks
    
    # =======================================================
    
    logger.info("="*60)
    logger.info("SageNet Philosophical Text Preprocessing Pipeline")
    logger.info("="*60)
    logger.info(f"PDF Directory: {Path(PDF_DIRECTORY).absolute()}")
    logger.info(f"Output File: {Path(OUTPUT_FILE).absolute()}")
    logger.info(f"Chunk Size: {CHUNK_SIZE} words")
    logger.info(f"Chunk Overlap: {CHUNK_OVERLAP} words")
    logger.info("="*60 + "\n")
    
    # Create preprocessor instance
    preprocessor = PhilosophicalTextPreprocessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process all books (this does everything!)
    preprocessor.process_all_books(PDF_DIRECTORY, OUTPUT_FILE)
    
    # ==================== OPTIONAL: View Sample ====================
    # Uncomment this to see a sample chunk after processing
    
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        logger.info("\n" + "="*60)
        logger.info("SAMPLE CHUNK (first one in the file):")
        logger.info("="*60)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                first_chunk = json.loads(first_line)
                logger.info(f"Philosopher: {first_chunk['philosopher']}")
                logger.info(f"Source: {first_chunk['source_book']}")
                logger.info(f"Page: {first_chunk['page_number']}")
                logger.info(f"Words: {first_chunk['word_count']}")
                logger.info(f"Chunk ID: {first_chunk['chunk_id']}")
                logger.info(f"\nText preview (first 400 characters):")
                logger.info("-" * 60)
                logger.info(first_chunk['text'][:400] + "...")
                logger.info("-" * 60)
        
        logger.info("\n✅ All done! Your corpus is ready for the next step.")
        logger.info(f"📁 Output file: {output_path.absolute()}")
    else:
        logger.error("❌ Output file was not created. Check errors above.")
