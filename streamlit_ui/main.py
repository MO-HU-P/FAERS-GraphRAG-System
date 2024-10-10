import os
import logging
import time
from typing import Dict, Any, Optional, List
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.prompts import PromptTemplate
import streamlit as st

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_neo4j_graph() -> Neo4jGraph:
    try:
        return Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
    except Exception as e:
        logging.error(f"Failed to initialize Neo4j graph: {e}")
        raise

def init_llm() -> ChatOpenAI:
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        raise

def run_query_with_error_handling(graph: Neo4jGraph, query: str) -> Optional[list]:
    try:
        return graph.query(query)
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return None

def check_embeddings_exist(graph: Neo4jGraph, node_labels: list) -> bool:
    for label in node_labels:
        query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NOT NULL
        RETURN COUNT(n) > 0 AS exists
        LIMIT 1
        """
        result = run_query_with_error_handling(graph, query)
        if result and result[0]['exists']:
            return True
    return False

# Define updated graph schema
schema = """
Node Properties:
- Drug(name: string, primarySubstabce: string)
- Case(primaryid: integer, age: float, ageUnit: string, gender: string, eventDate: date, reportDate: date, reporterOccupation: string)
- Reaction(description: string)
- Outcome(code: string, outcome: string)
- ReportSource(code: string, name: string)
- AgeGroup(ageGroup: string)
- Manufacturer(manufacturerName: string)
- Therapy(primaryid: integer)

Relationships:
- (Case)-[FALLS_UNDER]->(AgeGroup)
- (Manufacturer)-[REGISTERED]->(Case)
- (Case)-[RESULTED_IN]->(Outcome)
- (Case)-[HAS_REACTION]->(Reaction)
- (Case)-[REPORTED_BY]->(ReportSource)
- (Case)-[IS_PRIMARY_SUSPECT]->(Drug)
- (Case)-[IS_SECONDARY_SUSPECT]->(Drug)
- (Case)-[IS_CONCOMITANT]->(Drug)
- (Case)-[IS_INTERACTING]->(Drug)
- (Case)-[RECEIVED]->(Therapy)
- (Therapy)-[PRESCRIBED]->(Drug)

Relationship Properties:
- IS_PRIMARY_SUSPECT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_SECONDARY_SUSPECT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_CONCOMITANT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_INTERACTING(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- PRESCRIBED(drugSequence: integer, startYear: integer = 1900, endYear: integer = 2021)

Constraints:
- Drug.name is unique
- Case.primaryid is unique
- Reaction.description is unique
- ReportSource.code is unique
- Outcome.code is unique
- Therapy.primaryid is unique
- Manufacturer.manufacturerName is unique

Indexes:
- Case.age
- Case.ageUnit
- Case.gender
- Case.eventDate
- Case.reportDate

Node Descriptions:
- Case: Demographic information of a person (Case) involved in the adverse event report.
- Drug: Drug involved in the adverse event. Drug can be a primary suspect, secondary suspect, concomitant or interacting drug responsible for the adverse effect. This suspect type is identified by the relationship between Case and Drug Nodes.
- Reaction: Adverse reaction(side effect) that the Case developed after consumption of the respective Drug.
- Outcome: Long-term outcome of the Case after the Reaction.
- ReportSource: Information about who reported the adverse event.
- Therapy: Details about the therapy received by the case, including drug administration.
- AgeGroup: Age information reporting of case.

Relationship Descriptions:
- FALLS_UNDER: Relate Case and AgeGroup
- REGISTERED: Relate Case and Manufacturer
- RESULTED_IN: Relate Case and Outcome
- HAS_REACTION: Relate Case and Reaction
- REPORTED_BY: Relate Case and ReportSource
- IS_PRIMARY_SUSPECT: Relate Case and Drug based on primary suspect type
- IS_SECONDARY_SUSPECT: Relate Case and Drug based on secondary suspect type
- IS_CONCOMITANT: Relate Case and concomitant of Drug
- IS_INTERACTING: Relate Case and interacting among Drug
- RECEIVED: Relate Case and Therapy
- PRESCRIBED: Relate Therapy and Drug
"""

CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query the FDA Adverse Event Reporting System graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Optimize the query for performance by using appropriate indexes and limiting results where applicable.

Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

def create_graph_qa_chain(graph: Neo4jGraph, llm: ChatOpenAI) -> GraphCypherQAChain:
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT
    )

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

def create_neo4j_vectors(graph: Neo4jGraph) -> Dict[str, Neo4jVector]:
    try:
        embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model='nomic-embed-text')
        vector_stores = {}
        
        node_configs = {
            "Drug": ["name", "primarySubstabce"],
            "Reaction": ["description"],
            "Outcome": ["code", "outcome"],
            "Case": ["primaryid", "age", "ageUnit", "gender", "eventDate", "reportDate", "reporterOccupation"],
            "Therapy": ["primaryid"],
            "AgeGroup": ["ageGroup"],
            "ReportSource": ["code", "name"],
            "Manufacturer": ["manufacturerName"]
        }
        
        for label, properties in node_configs.items():
            vector_stores[label] = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                graph=graph,
                node_label=label,
                text_node_properties=properties,
                embedding_node_property="embedding",
                search_type="hybrid"
            )
        
        return vector_stores
    except Exception as e:
        logging.error(f"Error creating Neo4jVectors: {e}")
        return {}

def create_retrieval_qa_chains(vector_indices: Dict[str, Neo4jVector], llm: ChatOpenAI) -> Dict[str, RetrievalQA]:
    retrieval_qa_chains = {}
    for label, vector_index in vector_indices.items():
        retrieval_qa_chains[label] = RetrievalQA.from_chain_type(
            llm, 
            retriever=vector_index.as_retriever(), 
            verbose=True
        )
    return retrieval_qa_chains

final_answer_template = """
You are an AI assistant specializing in pharmaceutical adverse events. Use the following information to answer the user's question:

1. Graph Database Information:
{graph_result}

2. Vector Search Results (if available):
{vector_result}

3. Database Schema:
{schema}

Question: {question}

Please provide a comprehensive and accurate answer based on the available information. Follow these guidelines:
1. Identify relationships between Drug, Case, Reaction, Outcome, Therapy, AgeGroup, ReportSource, and Manufacturer using the graph structure.
2. Utilize vector search results to find similar cases or relevant information.
3. Use ALL available information from BOTH the graph database and vector search results, even if incomplete.
4. If either source provides partial information, use it to construct the best possible answer.
5. Combine and synthesize information from both sources to provide the most comprehensive answer possible.
6. If one source lacks information, rely more heavily on the other source to formulate your response.
7. Clearly state which source (graph or vector) the information comes from in your answer.
8. When information is uncertain or lacking, state it explicitly.
9. Suggest follow-up queries that could provide more detailed information on the topic.
10. Provide relevant statistical information (e.g., adverse event frequency).
11. Highlight patterns or trends suggested by the data.
12. Consider age, sex, and other patient characteristics in your analysis.
13. Prioritize patient safety, mentioning significant risks or warnings.
14. Discuss limitations in the data or analysis that may affect result interpretation.
15. If information is limited, speculate on possible answers based on available data, clearly labeling such speculation.

Answer:
"""

final_answer_prompt = PromptTemplate(
    input_variables=["graph_result", "vector_result", "schema", "question"],
    template=final_answer_template
)

def create_final_chain(llm: ChatOpenAI) -> Any:
    return final_answer_prompt | llm


def process_results(graph_result: Optional[Dict[str, Any]], vector_results: Optional[Dict[str, Any]]) -> Dict[str, str]:
    combined_result = {}
    
    if graph_result and 'result' in graph_result:
        combined_result['graph'] = graph_result['result']
    else:
        combined_result['graph'] = "No graph database results available."
    
    vector_result_str = ""
    if vector_results:
        for label, result in vector_results.items():
            if isinstance(result, dict) and 'result' in result:
                vector_result_str += f"{label}: {result['result']}\n"
            elif isinstance(result, str):
                vector_result_str += f"{label}: {result}\n"
            elif result is None:
                vector_result_str += f"{label}: Vector search failed or not available.\n"
    
    if vector_result_str:
        combined_result['vector'] = vector_result_str
    else:
        combined_result['vector'] = "No vector search results available."
    
    return combined_result

def answer_question(
    question: str,
    graph_qa: GraphCypherQAChain,
    retrieval_qa_chains: Optional[Dict[str, RetrievalQA]],
    final_chain: Any
) -> Dict[str, Any]:
    start_time = time.time()
    logging.info(f"Processing question: {question}")
    
    try:
        graph_result = graph_qa.invoke(question)
    except Exception as e:
        logging.error(f"Error in graph query: {e}")
        graph_result = {"result": "Graph query failed"}
    
    vector_results = None
    if retrieval_qa_chains:
        vector_results = {}
        for label, chain in retrieval_qa_chains.items():
            try:
                vector_results[label] = chain.invoke(question)
            except Exception as e:
                logging.error(f"Error in vector query for {label}: {e}")
                vector_results[label] = f"Vector query for {label} failed"
    
    combined_results = process_results(graph_result, vector_results)
    
    try:
        final_answer = final_chain.invoke({
            "graph_result": combined_results['graph'],
            "vector_result": combined_results['vector'],
            "schema": schema,
            "question": question
        })
    except Exception as e:
        logging.error(f"Error in final answer generation: {e}")
        final_answer = {"content": "Failed to generate final answer"}
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "answer": final_answer.content if hasattr(final_answer, 'content') else str(final_answer),
        "processing_time": processing_time,
        "graph_result": combined_results['graph'],
        "vector_result": combined_results['vector']
    }

def main():

    st.title("GraphRAG system for FAERS data")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    try:
        graph = init_neo4j_graph()
        llm = init_llm()
        
        graph_qa = create_graph_qa_chain(graph, llm)
        
        retrieval_qa_chains = {}
        try:
            vector_indices = create_neo4j_vectors(graph)
            if vector_indices:
                retrieval_qa_chains = create_retrieval_qa_chains(vector_indices, llm)
                logging.info(f"Vector indices created successfully for {', '.join(vector_indices.keys())}. Using hybrid retrieval with RetrievalQA.")
            else:
                logging.warning("No vector indices were created. Using GraphCypherQAChain only.")
        except Exception as e:
            logging.error(f"Error creating vector indices: {e}. Using GraphCypherQAChain only.")

        final_chain = create_final_chain(llm)
        
        if st.button("Clear History"):
            st.session_state.messages = []
            st.experimental_rerun()

        if question := st.chat_input("Enter your question"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})

            result = answer_question(question, graph_qa, retrieval_qa_chains, final_chain)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(f"\nAnswer: {result['answer']}")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result['answer']})

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()





