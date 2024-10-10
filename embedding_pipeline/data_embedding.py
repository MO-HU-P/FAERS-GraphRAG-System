import os
import logging
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j graph
logging.info("Initializing Neo4j graph")
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')


# Initialize Embeddings model
logging.info("Initializing Embeddings model")
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def run_query(query, params=None):
    try:
        result = graph.query(query, params=params)
        return result
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        raise

def get_embedding(text):
    try:
        preprocessed_text = preprocess_text(text)
        return embeddings.embed_query(preprocessed_text)
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        raise

def process_batch(batch, entity_type):
    embeddings_batch = []
    for entity in batch:
        try:
            if entity_type == 'Drug':
                entity_id = entity['n.name']
                text = f"Drug: {entity.get('n.name', '')} {entity.get('n.primarySubstabce', '')}" 
            elif entity_type == 'Therapy':
                entity_id = entity['n.primaryid']
                text = f"Therapy: {entity.get('n.primaryid', '')}"
            elif entity_type == 'Reaction':
                entity_id = entity['n.description']
                text = f"Reaction: {entity.get('n.description', '')}"
            elif entity_type == 'Case':
                entity_id = entity['n.primaryid']
                text = f"Case: {entity.get('n.primaryid', '')} {entity.get('n.age', '')} {entity.get('n.ageUnit', '')} {entity.get('n.gender', '')} {entity.get('n.eventDate', '')} {entity.get('n.reportDate', '')} {entity.get('n.reporterOccupation', '')}"
            elif entity_type == 'Outcome':
                entity_id = entity['n.code']
                text = f"Outcome: {entity.get('n.code', '')} {entity.get('n.outcome', '')}"
            elif entity_type == 'ReportSource':
                entity_id = entity['n.code']
                text = f"ReportSource: {entity.get('n.code', '')} {entity.get('n.name', '')}"
            elif entity_type == 'AgeGroup':
                entity_id = entity['n.ageGroup']
                text = f"AgeGroup: {entity.get('n.ageGroup', '')}"
            elif entity_type == 'Manufacturer':
                entity_id = entity['n.manufacturerName']
                text = f"Manufacturer: {entity.get('n.manufacturerName', '')}"

            embedding = get_embedding(text)
            embeddings_batch.append((entity_id, embedding))
        except Exception as e:
            logging.error(f"Error processing entity {entity_id} of type {entity_type}: {str(e)}")
    
    return embeddings_batch

def process_batch_with_retry(batch, entity_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            return process_batch(batch, entity_type)
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed for {entity_type}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.error(f"All retry attempts failed for {entity_type} batch.")
                logging.error(f"Error: {str(e)}")
                raise

def add_embeddings(entity_type, cypher_query, update_query):
    logging.info(f"Adding embeddings to {entity_type}")
    
    results = run_query(cypher_query)
    
    if not results:
        logging.info(f"All {entity_type} have already been processed. Skipping.")
        return
    
    batch_size = 100
    batches = [results[i:i + batch_size] for i in range(0, len(results), batch_size)]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_batch_with_retry, batch, entity_type) for batch in batches]
        
        with tqdm(total=len(results), desc=f"Processing {entity_type}") as pbar:
            for future in as_completed(futures):
                try:
                    embeddings_batch = future.result()
                    
                    update_params = {'batch': [{'id': id, 'embedding': emb} for id, emb in embeddings_batch]}
                    run_query(update_query, update_params)
                    
                    pbar.update(len(embeddings_batch))
                except Exception as e:
                    logging.error(f"Error processing batch for {entity_type}: {str(e)}")
    
    logging.info(f"Processed {len(results)} {entity_type}")

def add_embeddings_to_nodes():
    node_types = {
        'Drug': {
            'query': """
            MATCH (n:Drug)
            WHERE n.embedding IS NULL
            RETURN n.name, n.primarySubstabce
            ORDER BY n.name
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Drug {name: item.id})
            SET n.embedding = item.embedding
            """
        },
        'Therapy': {
            'query': """
            MATCH (n:Therapy)
            WHERE n.embedding IS NULL
            RETURN n.primaryid
            ORDER BY n.primaryid
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Therapy {primaryid: toInteger(item.id)})
            SET n.embedding = item.embedding
            """
        },
        'Reaction': {
            'query': """
            MATCH (n:Reaction)
            WHERE n.embedding IS NULL
            RETURN n.description
            ORDER BY n.description
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Reaction {description: item.id})
            SET n.embedding = item.embedding
            """
        },
        'Case': {
            'query': """
            MATCH (n:Case)
            WHERE n.embedding IS NULL
            RETURN n.primaryid, n.age, n.ageUnit, n.gender, n.eventDate, n.reportDate, n.reporterOccupation
            ORDER BY n.primaryid
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Case {primaryid: item.id})
            SET n.embedding = item.embedding
            """
        },
        'Outcome': {
            'query': """
            MATCH (n:Outcome)
            WHERE n.embedding IS NULL
            RETURN n.code, n.outcome
            ORDER BY n.code
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Outcome {code: item.id})
            SET n.embedding = item.embedding
            """
        },
        'ReportSource': {
            'query': """
            MATCH (n:ReportSource)
            WHERE n.embedding IS NULL
            RETURN n.code, n.name
            ORDER BY n.code
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:ReportSource {code: item.id})
            SET n.embedding = item.embedding
            """
        },
        'AgeGroup': {
            'query': """
            MATCH (n:AgeGroup)
            WHERE n.embedding IS NULL
            RETURN n.ageGroup
            ORDER BY n.ageGroup
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:AgeGroup {ageGroup: item.id})
            SET n.embedding = item.embedding
            """
        },
        'Manufacturer': {
            'query': """
            MATCH (n:Manufacturer)
            WHERE n.embedding IS NULL
            RETURN n.manufacturerName
            ORDER BY n.manufacturerName
            """,
            'update': """
            UNWIND $batch AS item
            MATCH (n:Manufacturer {manufacturerName: item.id})
            SET n.embedding = item.embedding
            """
        }
    }
    
    for node_type, queries in node_types.items():
        add_embeddings(node_type, queries['query'], queries['update'])

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        graph.refresh_schema()
        logging.info(f"Graph schema: {graph.schema}")
        
        add_embeddings_to_nodes()
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"All embeddings added successfully. Total execution time: {total_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())