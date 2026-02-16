"""
SageNet - Enhanced Knowledge Graph Construction
===============================================
Builds a comprehensive Neo4j knowledge graph with rich relationships.

Enhanced Graph Structure:
-------------------------
NODES:
- Philosopher (5): name, era, birth_year, death_year, school
- School (5): name, era, description
- Concept (1000+): name, frequency, category
- TextChunk (6000+): id, text, section, word_count, page_number

RELATIONSHIPS:
- WROTE: Philosopher → TextChunk (authorship)
- BELONGS_TO: Philosopher → School (tradition)
- DISCUSSES: TextChunk → Concept (mentions concept)
- EXPLORED: Philosopher → Concept (aggregated concept exploration)
- PRECEDED: Philosopher → Philosopher (temporal/historical)
- SHARES_CONCEPT_WITH: Philosopher → Philosopher (common ideas)
- RELATED_TO: Concept → Concept (co-occurrence)
- INFLUENCED: Philosopher → Philosopher (intellectual influence)
"""

from neo4j import GraphDatabase
import json
from pathlib import Path
import logging
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import re
from itertools import combinations
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPhilosophicalKG:
    """
    Enhanced Knowledge Graph with rich relationships and detailed nodes.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        logger.info(f"Connecting to Neo4j at {uri}...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            logger.info("✅ Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f" Failed to connect: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing data."""
        logger.info("Clearing existing data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_constraints(self):
        """Create constraints and indexes for performance."""
        logger.info("Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT philosopher_name IF NOT EXISTS FOR (p:Philosopher) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT school_name IF NOT EXISTS FOR (s:School) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.id IS UNIQUE",
            # Create indexes for better query performance
            "CREATE INDEX philosopher_era IF NOT EXISTS FOR (p:Philosopher) ON (p.era)",
            "CREATE INDEX concept_frequency IF NOT EXISTS FOR (c:Concept) ON (c.frequency)",
            "CREATE INDEX chunk_philosopher IF NOT EXISTS FOR (t:TextChunk) ON (t.philosopher)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass
        
        logger.info("Constraints and indexes created")
    
    def create_philosopher_nodes(self, chunks: List[Dict]):
        """Create enhanced Philosopher nodes."""
        logger.info("Creating Philosopher nodes with metadata...")
        
        philosopher_metadata = {
            'Aristotle': {
                'birth_year': -384,
                'death_year': -322,
                'era': 'Ancient Greece',
                'school': 'Peripatetic',
                'description': 'Student of Plato, tutor of Alexander the Great, founder of the Lyceum'
            },
            'Plato': {
                'birth_year': -428,
                'death_year': -348,
                'era': 'Ancient Greece',
                'school': 'Platonism',
                'description': 'Student of Socrates, founded the Academy, developed Theory of Forms'
            },
            'Buddha': {
                'birth_year': -563,
                'death_year': -483,
                'era': 'Ancient India',
                'school': 'Buddhism',
                'description': 'Founded Buddhism, taught the Four Noble Truths and Eightfold Path'
            },
            'Immanuel Kant': {
                'birth_year': 1724,
                'death_year': 1804,
                'era': 'Enlightenment',
                'school': 'German Idealism',
                'description': 'Critical philosophy, categorical imperative, transcendental idealism'
            },
            'Confucius': {
                'birth_year': -551,
                'death_year': -479,
                'era': 'Ancient China',
                'school': 'Confucianism',
                'description': 'Founded Confucianism, emphasized moral cultivation and social harmony'
            }
        }
        
        chunk_counts = Counter(chunk['philosopher'] for chunk in chunks)
        
        with self.driver.session() as session:
            for phil_name, metadata in philosopher_metadata.items():
                session.run("""
                    CREATE (p:Philosopher {
                        name: $name,
                        birth_year: $birth_year,
                        death_year: $death_year,
                        era: $era,
                        school: $school,
                        description: $description,
                        chunk_count: $chunk_count,
                        lifespan_years: $lifespan
                    })
                """,
                    name=phil_name,
                    birth_year=metadata['birth_year'],
                    death_year=metadata['death_year'],
                    era=metadata['era'],
                    school=metadata['school'],
                    description=metadata['description'],
                    chunk_count=chunk_counts.get(phil_name, 0),
                    lifespan=metadata['death_year'] - metadata['birth_year']
                )
        
        logger.info("Created 5 Philosopher nodes")
    
    def create_school_nodes(self):
        """Create School of Thought nodes."""
        logger.info("Creating School nodes...")
        
        schools = [
            {
                'name': 'Ancient Greek Philosophy',
                'era': 'Ancient',
                'description': 'Philosophy of ancient Greece, emphasizing rational inquiry',
                'key_concepts': 'Forms, Ethics, Politics, Logic'
            },
            {
                'name': 'Buddhism',
                'era': 'Ancient',
                'description': 'Eastern philosophy focused on suffering, enlightenment, and compassion',
                'key_concepts': 'Four Noble Truths, Eightfold Path, Nirvana, Dharma'
            },
            {
                'name': 'Confucianism',
                'era': 'Ancient',
                'description': 'Chinese philosophy emphasizing moral cultivation and social harmony',
                'key_concepts': 'Ren (benevolence), Li (ritual), Filial Piety'
            },
            {
                'name': 'German Idealism',
                'era': 'Modern',
                'description': 'Post-Kantian philosophy emphasizing the role of mind in reality',
                'key_concepts': 'Transcendental Idealism, Categorical Imperative, Autonomy'
            },
            {
                'name': 'Peripatetic',
                'era': 'Ancient',
                'description': 'Aristotelian school focusing on empirical observation and logic',
                'key_concepts': 'Substance, Causation, Virtue Ethics, Golden Mean'
            }
        ]
        
        with self.driver.session() as session:
            for school in schools:
                session.run("""
                    CREATE (s:School {
                        name: $name,
                        era: $era,
                        description: $description,
                        key_concepts: $key_concepts
                    })
                """, **school)
        
        logger.info("Created 5 School nodes")
    
    def extract_concepts_advanced(self, chunks: List[Dict], min_frequency: int = 5) -> Dict[str, Dict]:
        """
        Advanced concept extraction with frequency and categorization.
        
        Returns:
            Dict mapping concept name to metadata (frequency, category, chunks)
        """
        logger.info("Extracting concepts with advanced NLP...")
        
        # Expanded philosophical concept list with categories
        concept_categories = {
            # Ethics & Morality
            'virtue': 'ethics', 'ethics': 'ethics', 'morality': 'ethics', 
            'justice': 'ethics', 'duty': 'ethics', 'right': 'ethics', 'wrong': 'ethics',
            'good': 'ethics', 'evil': 'ethics', 'moral': 'ethics',
            
            # Epistemology
            'knowledge': 'epistemology', 'truth': 'epistemology', 'belief': 'epistemology',
            'wisdom': 'epistemology', 'understanding': 'epistemology', 'reason': 'epistemology',
            'logic': 'epistemology', 'doubt': 'epistemology', 'certainty': 'epistemology',
            
            # Metaphysics
            'being': 'metaphysics', 'existence': 'metaphysics', 'reality': 'metaphysics',
            'essence': 'metaphysics', 'substance': 'metaphysics', 'causation': 'metaphysics',
            'time': 'metaphysics', 'space': 'metaphysics', 'nature': 'metaphysics',
            
            # Mind & Consciousness
            'soul': 'mind', 'mind': 'mind', 'consciousness': 'mind',
            'self': 'mind', 'identity': 'mind', 'perception': 'mind',
            'thought': 'mind', 'awareness': 'mind', 'spirit': 'mind',
            
            # Buddhist Concepts
            'suffering': 'buddhist', 'enlightenment': 'buddhist', 'nirvana': 'buddhist',
            'dharma': 'buddhist', 'karma': 'buddhist', 'compassion': 'buddhist',
            'meditation': 'buddhist', 'mindfulness': 'buddhist',
            
            # Political Philosophy
            'freedom': 'political', 'power': 'political', 'authority': 'political',
            'law': 'political', 'state': 'political', 'society': 'political',
            'government': 'political', 'citizen': 'political',
            
            # Aesthetics
            'beauty': 'aesthetics', 'art': 'aesthetics', 'taste': 'aesthetics',
            
            # Logic & Reasoning
            'argument': 'logic', 'proof': 'logic', 'inference': 'logic',
            
            # Kantian Concepts
            'categorical imperative': 'kantian', 'autonomy': 'kantian',
            'practical reason': 'kantian', 'transcendental': 'kantian',
            
            # Platonic Concepts
            'forms': 'platonic', 'ideas': 'platonic', 'ideal': 'platonic',
            
            # Confucian Concepts
            'filial piety': 'confucian', 'ritual': 'confucian', 'benevolence': 'confucian',
            'propriety': 'confucian', 'righteousness': 'confucian',
            
            # Other
            'happiness': 'other', 'pleasure': 'other', 'pain': 'other',
            'desire': 'other', 'will': 'other', 'emotion': 'other'
        }
        
        concept_data = defaultdict(lambda: {
            'frequency': 0,
            'category': 'other',
            'chunk_ids': set(),
            'philosophers': set()
        })
        
        # Count concept occurrences
        for chunk in tqdm(chunks, desc="Analyzing concepts"):
            text_lower = chunk['text'].lower()
            chunk_id = chunk['chunk_id']
            philosopher = chunk['philosopher']
            
            for concept, category in concept_categories.items():
                pattern = r'\b' + re.escape(concept) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                
                if matches > 0:
                    concept_data[concept]['frequency'] += matches
                    concept_data[concept]['category'] = category
                    concept_data[concept]['chunk_ids'].add(chunk_id)
                    concept_data[concept]['philosophers'].add(philosopher)
        
        # Filter by minimum frequency
        filtered_concepts = {
            name: data for name, data in concept_data.items()
            if data['frequency'] >= min_frequency
        }
        
        logger.info(f"✅ Extracted {len(filtered_concepts)} concepts (min frequency: {min_frequency})")
        return filtered_concepts
    
    def create_concept_nodes(self, concept_data: Dict):
        """Create Concept nodes with metadata."""
        logger.info("Creating Concept nodes...")
        
        with self.driver.session() as session:
            for concept_name, data in tqdm(concept_data.items(), desc="Creating concepts"):
                session.run("""
                    CREATE (c:Concept {
                        name: $name,
                        frequency: $frequency,
                        category: $category,
                        philosopher_count: $philosopher_count
                    })
                """,
                    name=concept_name.title(),
                    frequency=data['frequency'],
                    category=data['category'],
                    philosopher_count=len(data['philosophers'])
                )
        
        logger.info(f" Created {len(concept_data)} Concept nodes")
    
    def create_textchunk_nodes(self, chunks: List[Dict], batch_size: int = 500):
        """Create TextChunk nodes."""
        logger.info("Creating TextChunk nodes...")
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Creating chunks"):
                batch = chunks[i:i + batch_size]
                
                session.run("""
                    UNWIND $chunks as chunk
                    CREATE (t:TextChunk {
                        id: chunk.chunk_id,
                        text: chunk.text,
                        philosopher: chunk.philosopher,
                        source_book: chunk.source_book,
                        page_number: chunk.page_number,
                        chunk_index: chunk.chunk_index,
                        word_count: chunk.word_count
                    })
                """, chunks=batch)
        
        logger.info(f" Created {len(chunks)} TextChunk nodes")
    
    def create_wrote_relationships(self):
        """Create WROTE relationships: Philosopher → TextChunk."""
        logger.info("Creating WROTE relationships...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Philosopher), (t:TextChunk)
                WHERE p.name = t.philosopher
                CREATE (p)-[:WROTE]->(t)
                RETURN count(*) as count
            """)
            count = result.single()['count']
        
        logger.info(f"Created {count:,} WROTE relationships")
    
    def create_belongs_to_relationships(self):
        """Create BELONGS_TO relationships: Philosopher → School."""
        logger.info("Creating BELONGS_TO relationships...")
        
        school_mapping = {
            'Aristotle': 'Peripatetic',
            'Plato': 'Ancient Greek Philosophy',
            'Buddha': 'Buddhism',
            'Confucius': 'Confucianism',
            'Immanuel Kant': 'German Idealism'
        }
        
        with self.driver.session() as session:
            for philosopher, school in school_mapping.items():
                session.run("""
                    MATCH (p:Philosopher {name: $philosopher})
                    MATCH (s:School {name: $school})
                    CREATE (p)-[:BELONGS_TO]->(s)
                """, philosopher=philosopher, school=school)
        
        logger.info(f" Created {len(school_mapping)} BELONGS_TO relationships")
    
    def create_discusses_relationships(self, concept_data: Dict, batch_size: int = 1000):
        """Create DISCUSSES relationships: TextChunk → Concept."""
        logger.info("Creating DISCUSSES relationships...")
        
        relationships = []
        for concept_name, data in concept_data.items():
            for chunk_id in data['chunk_ids']:
                relationships.append({
                    'chunk_id': chunk_id,
                    'concept': concept_name.title()
                })
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(relationships), batch_size), desc="DISCUSSES"):
                batch = relationships[i:i + batch_size]
                session.run("""
                    UNWIND $rels as rel
                    MATCH (t:TextChunk {id: rel.chunk_id})
                    MATCH (c:Concept {name: rel.concept})
                    CREATE (t)-[:DISCUSSES]->(c)
                """, rels=batch)
        
        logger.info(f" Created {len(relationships):,} DISCUSSES relationships")
    
    def create_explored_relationships(self, concept_data: Dict):
        """Create EXPLORED relationships: Philosopher → Concept (aggregated)."""
        logger.info("Creating EXPLORED relationships...")
        
        with self.driver.session() as session:
            for concept_name, data in tqdm(concept_data.items(), desc="EXPLORED"):
                for philosopher in data['philosophers']:
                    # Count how many times this philosopher discusses this concept
                    session.run("""
                        MATCH (p:Philosopher {name: $philosopher})
                        MATCH (c:Concept {name: $concept})
                        MATCH (p)-[:WROTE]->(t:TextChunk)-[:DISCUSSES]->(c)
                        WITH p, c, count(t) as frequency
                        CREATE (p)-[:EXPLORED {frequency: frequency}]->(c)
                    """,
                        philosopher=philosopher,
                        concept=concept_name.title()
                    )
        
        logger.info(" Created EXPLORED relationships")
    
    def create_preceded_relationships(self):
        """Create PRECEDED relationships: Philosopher → Philosopher (temporal)."""
        logger.info("Creating PRECEDED relationships...")
        
        with self.driver.session() as session:
            # Create PRECEDED for all philosophers where one died before the other was born
            session.run("""
                MATCH (p1:Philosopher), (p2:Philosopher)
                WHERE p1.death_year < p2.birth_year
                WITH p1, p2, p2.birth_year - p1.death_year as years_apart
                CREATE (p1)-[:PRECEDED {years_apart: years_apart}]->(p2)
            """)
            
            # Count
            result = session.run("""
                MATCH (p1)-[r:PRECEDED]->(p2)
                RETURN count(r) as count
            """)
            count = result.single()['count']
        
        logger.info(f"Created {count} PRECEDED relationships")
    
    def create_shares_concept_relationships(self):
        """Create SHARES_CONCEPT_WITH relationships: Philosopher ↔ Philosopher."""
        logger.info("Creating SHARES_CONCEPT_WITH relationships...")
        
        with self.driver.session() as session:
            session.run("""
                MATCH (p1:Philosopher)-[:EXPLORED]->(c:Concept)<-[:EXPLORED]-(p2:Philosopher)
                WHERE id(p1) < id(p2)
                WITH p1, p2, collect(c.name) as shared_concepts, count(c) as concept_count
                CREATE (p1)-[:SHARES_CONCEPT_WITH {
                    shared_concepts: shared_concepts,
                    count: concept_count
                }]->(p2)
            """)
            
            result = session.run("""
                MATCH (p1)-[r:SHARES_CONCEPT_WITH]->(p2)
                RETURN count(r) as count
            """)
            count = result.single()['count']
        
        logger.info(f"Created {count} SHARES_CONCEPT_WITH relationships")
    
    def create_related_to_relationships(self, min_cooccurrence: int = 5):
        """Create RELATED_TO relationships: Concept ↔ Concept (co-occurrence)."""
        logger.info("Creating RELATED_TO relationships...")
        
        with self.driver.session() as session:
            # Find concepts that appear together in same chunks
            session.run("""
                MATCH (t:TextChunk)-[:DISCUSSES]->(c1:Concept)
                MATCH (t)-[:DISCUSSES]->(c2:Concept)
                WHERE id(c1) < id(c2)
                WITH c1, c2, count(t) as strength
                WHERE strength >= $min_cooccurrence
                CREATE (c1)-[:RELATED_TO {strength: strength}]->(c2)
            """, min_cooccurrence=min_cooccurrence)
            
            result = session.run("""
                MATCH (c1)-[r:RELATED_TO]->(c2)
                RETURN count(r) as count
            """)
            count = result.single()['count']
        
        logger.info(f"Created {count} RELATED_TO relationships (min co-occurrence: {min_cooccurrence})")
    
    def create_influenced_relationships(self):
        """Create INFLUENCED relationships (historical/intellectual influence)."""
        logger.info("Creating INFLUENCED relationships...")
        
        influences = [
            ('Plato', 'Aristotle', 'direct', 'Plato was Aristotle\'s teacher'),
            ('Aristotle', 'Immanuel Kant', 'historical', 'Kant built on Aristotelian metaphysics'),
            ('Plato', 'Immanuel Kant', 'historical', 'Platonic idealism influenced Kant'),
        ]
        
        with self.driver.session() as session:
            for influencer, influenced, influence_type, description in influences:
                session.run("""
                    MATCH (p1:Philosopher {name: $influencer})
                    MATCH (p2:Philosopher {name: $influenced})
                    CREATE (p1)-[:INFLUENCED {
                        type: $type,
                        description: $description
                    }]->(p2)
                """,
                    influencer=influencer,
                    influenced=influenced,
                    type=influence_type,
                    description=description
                )
        
        logger.info(f"Created {len(influences)} INFLUENCED relationships")
    
    def get_statistics(self):
        """Print comprehensive graph statistics."""
        logger.info("\n" + "="*80)
        logger.info("ENHANCED KNOWLEDGE GRAPH STATISTICS")
        logger.info("="*80)
        
        with self.driver.session() as session:
            # Node counts
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """).data()
            
            logger.info("\n📊 NODE COUNTS:")
            total_nodes = 0
            for item in node_counts:
                count = item['count']
                total_nodes += count
                logger.info(f"  {item['label']:20s}: {count:>8,}")
            logger.info(f"  {'TOTAL':20s}: {total_nodes:>8,}")
            
            # Relationship counts
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            logger.info("\n🔗 RELATIONSHIP COUNTS:")
            total_rels = 0
            for item in rel_counts:
                count = item['count']
                total_rels += count
                logger.info(f"  {item['type']:30s}: {count:>8,}")
            logger.info(f"  {'TOTAL':30s}: {total_rels:>8,}")
            
            # Top concepts
            top_concepts = session.run("""
                MATCH (c:Concept)
                RETURN c.name as concept, c.frequency as frequency, c.category as category
                ORDER BY c.frequency DESC
                LIMIT 10
            """).data()
            
            logger.info("\nTOP 10 CONCEPTS BY FREQUENCY:")
            for item in top_concepts:
                logger.info(f"  {item['concept']:20s} ({item['category']:15s}): {item['frequency']:>6,}")
        
        logger.info("="*80 + "\n")
    
    def example_queries(self):
        """Run example queries."""
        logger.info("\n" + "="*80)
        logger.info("EXAMPLE GRAPH QUERIES")
        logger.info("="*80)
        
        queries = [
            {
                'name': 'Find philosophers who share the most concepts',
                'cypher': """
                    MATCH (p1)-[r:SHARES_CONCEPT_WITH]->(p2)
                    RETURN p1.name as philosopher1, p2.name as philosopher2, 
                           r.count as shared_count
                    ORDER BY r.count DESC
                    LIMIT 5
                """
            },
            {
                'name': 'Find most discussed concepts by Aristotle',
                'cypher': """
                    MATCH (p:Philosopher {name: 'Aristotle'})-[r:EXPLORED]->(c:Concept)
                    RETURN c.name as concept, r.frequency as frequency
                    ORDER BY r.frequency DESC
                    LIMIT 5
                """
            },
            {
                'name': 'Find strongly related concepts',
                'cypher': """
                    MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
                    RETURN c1.name as concept1, c2.name as concept2, r.strength as strength
                    ORDER BY r.strength DESC
                    LIMIT 5
                """
            },
            {
                'name': 'Find temporal sequence of philosophers',
                'cypher': """
                    MATCH (p1)-[r:PRECEDED]->(p2)
                    RETURN p1.name as earlier, p2.name as later, r.years_apart as years_apart
                    ORDER BY r.years_apart
                    LIMIT 5
                """
            }
        ]
        
        with self.driver.session() as session:
            for query_info in queries:
                logger.info(f"\n {query_info['name']}:")
                logger.info("-"*80)
                results = session.run(query_info['cypher']).data()
                for result in results:
                    logger.info(f"  {result}")
        
        logger.info("\n" + "="*80 + "\n")


def main():
    """Main execution."""
    # Import configuration from .env file
    from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    
    CORPUS_PATH = "../data/philosophical_corpus.jsonl"
    
    logger.info("="*80)
    logger.info("SageNet Enhanced Knowledge Graph Construction")
    logger.info("="*80)
    logger.info(f"Corpus: {Path(CORPUS_PATH).absolute()}")
    logger.info("="*80 + "\n")
    
    # Load corpus
    logger.info("Loading corpus...")
    chunks = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks):,} chunks\n")
    
    # Initialize KG
    kg = EnhancedPhilosophicalKG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    try:
        # Build graph
        kg.clear_database()
        kg.create_constraints()
        
        # Create nodes
        kg.create_philosopher_nodes(chunks)
        kg.create_school_nodes()
        
        # Extract concepts
        concept_data = kg.extract_concepts_advanced(chunks, min_frequency=5)
        kg.create_concept_nodes(concept_data)
        
        # Create text chunks
        kg.create_textchunk_nodes(chunks)
        
        # Create relationships
        kg.create_wrote_relationships()
        kg.create_belongs_to_relationships()
        kg.create_discusses_relationships(concept_data)
        kg.create_explored_relationships(concept_data)
        kg.create_preceded_relationships()
        kg.create_shares_concept_relationships()
        kg.create_related_to_relationships(min_cooccurrence=5)
        kg.create_influenced_relationships()
        
        # Statistics
        kg.get_statistics()
        kg.example_queries()
        
        logger.info("="*80)
        logger.info("KNOWLEDGE GRAPH COMPLETE!")
        
        logger.info("="*80 + "\n")
        
    finally:
        kg.close()


if __name__ == "__main__":
    main()