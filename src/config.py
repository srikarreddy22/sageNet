"""
SageNet Configuration
Loads configuration from .env file and tests connections
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# LLM API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD not found in .env file\n"
        "Create a .env file in project root with:\n"
        "NEO4J_PASSWORD=your_password"
    )


def test_neo4j_connection():
    from neo4j import GraphDatabase
    

    print("TESTING NEO4J CONNECTION")

    print(f"URI:      {NEO4J_URI}")
    print(f"Username: {NEO4J_USERNAME}")
    print(f"Password: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'NOT SET'}")

    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
            if test_value == 1:
                print("Connection successful\n")
                
                version_result = session.run(
                    "CALL dbms.components() YIELD versions "
                    "RETURN versions[0] as version"
                )
                version = version_result.single()["version"]
                print(f"Neo4j Version: {version}")
                
                db_result = session.run("CALL db.info() YIELD name")
                db_name = db_result.single()["name"]
                print(f"Database Name: {db_name}")
                
                count_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = count_result.single()["count"]
                print(f"Current Nodes: {node_count:,}")
                
                print("\nNeo4j ready")
                return True
                
        driver.close()
        
    except Exception as e:
        print(f"\nConnection failed: {e}\n")
        print("Troubleshooting:")
        print("1. Is Neo4j running?")
        print("2. Is password correct in .env?")
        print("3. Is URI correct? (bolt://localhost:7687)")
        return False


def test_gemini():
    """Test Gemini API"""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in .env")
        return False
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Say 'test successful' in one word"
        )
        
        if response.text:
            print("Gemini working")
            return True
        return False
    except ImportError:
        print("Gemini library not installed: pip install google-genai")
        return False
    except Exception as e:
        print(f"Gemini error: {e}")
        return False


def test_groq():
    """Test Groq API"""
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found in .env")
        return False
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'test' in one word"}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("Groq working")
            return True
        return False
    except ImportError:
        print("Groq library not installed: pip install groq")
        return False
    except Exception as e:
        print(f"Groq error: {e}")
        return False


def test_all_llms():
    """Test all available LLM providers"""
    print("\n" + "="*70)
    print("TESTING LLM PROVIDERS")
    print("="*70)
    
    gemini_ok = test_gemini()
    groq_ok = test_groq()
    
    print("="*70)
    
    if gemini_ok or groq_ok:
        print("At least one LLM available")
        return True
    else:
        print("No LLM available - add API keys to .env")
        return False


def test_all_connections():
    """Test all system connections"""
    print("\nSageNet System Check")
    print("="*70 + "\n")
    
    neo4j_ok = test_neo4j_connection()
    llm_ok = test_all_llms()
    
    print("\n" + "="*70)
    print("SYSTEM STATUS")
    print("="*70)
    print(f"Neo4j:  {'OK' if neo4j_ok else 'FAILED'}")
    print(f"LLMs:   {'OK' if llm_ok else 'FAILED'}")
    print("="*70 + "\n")
    
    return neo4j_ok and llm_ok


if __name__ == "__main__":
    test_all_connections()