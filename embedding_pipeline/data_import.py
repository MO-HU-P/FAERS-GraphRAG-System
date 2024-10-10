import os
import logging
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.info("Initializing Neo4j graph")
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

def run_query(query):
    result = graph.query(query)
    logging.info(f"Query executed successfully: {query[:50]}...")
    return result

def import_data():
    import_queries = [
        # Constraints
        "CREATE CONSTRAINT constraint_drug_name IF NOT EXISTS FOR (n: `Drug`) REQUIRE n.`name` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_case_primaryid IF NOT EXISTS FOR (n: `Case`) REQUIRE n.`primaryid` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_reaction_description IF NOT EXISTS FOR (n: `Reaction`) REQUIRE n.`description` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_reportsource_code IF NOT EXISTS FOR (n: `ReportSource`) REQUIRE n.`code` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_outcome_code IF NOT EXISTS FOR (n: `Outcome`) REQUIRE n.`code` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_therapy_primaryid IF NOT EXISTS FOR (n: `Therapy`) REQUIRE n.`primaryid` IS UNIQUE;",
        "CREATE CONSTRAINT constraint_manufacturer_name IF NOT EXISTS FOR (n: `Manufacturer`) REQUIRE n.`manufacturerName` IS UNIQUE;",
        
        # Indexes
        "CREATE INDEX index_case_age IF NOT EXISTS FOR (n: `Case`) ON (n.`age`);",
        "CREATE INDEX index_case_ageUnit IF NOT EXISTS FOR (n: `Case`) ON (n.`ageUnit`);",
        "CREATE INDEX index_case_gender IF NOT EXISTS FOR (n: `Case`) ON (n.`gender`);",
        "CREATE INDEX index_case_eventdate IF NOT EXISTS FOR (n: `Case`) ON (n.`eventDate`);",
        "CREATE INDEX index_case_reportdate IF NOT EXISTS FOR (n: `Case`) ON (n.`reportDate`);",
        
        # Data import queries
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/demographics.csv" AS row
        MERGE (c:Case { primaryid: toInteger(row.primaryid) })
        ON CREATE SET
        c.eventDate= date(row.eventDate),
        c.reportDate= date(row.reportDate),
        c.age = toFloat(row.age),
        c.ageUnit = row.ageUnit,
        c.gender = row.sex,
        c.reporterOccupation = row.reporterOccupation
        MERGE (m:Manufacturer { manufacturerName: row.manufacturerName } )
        MERGE (m)-[:REGISTERED]->(c)
        MERGE (a:AgeGroup { ageGroup: row.ageGroup })
        MERGE (c)-[:FALLS_UNDER]->(a)
        RETURN count (c);
        """,
        
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/outcome.csv" AS row
        MERGE (o:Outcome { code: row.code })
        ON CREATE SET
        o.outcome = row.outcome
        WITH o, row
        MATCH (c:Case {primaryid: toInteger(row.primaryid)})
        MERGE (c)-[:RESULTED_IN]->(o)
        RETURN count(o);
        """,
        
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/reaction.csv" AS row
        MERGE (r:Reaction { description: row.description })
        WITH r, row
        MATCH (c:Case {primaryid: toInteger(row.primaryid)})
        MERGE (c)-[:HAS_REACTION]->(r)
        RETURN count(r);
        """,
        
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/reportSources.csv" AS row
        MERGE (r:ReportSource { code: row.code })
        ON CREATE SET
        r.name = row.name
        WITH r, row
        MATCH (c:Case {primaryid: toInteger(row.primaryid) })
        WITH c, r
        MERGE (c)-[:REPORTED_BY]->(r)
        RETURN count(r);
        """,
        
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/drugs-indication.csv" AS row
        MERGE (d:Drug { name: row.name })
        ON CREATE SET
        d.primarySubstabce = row.primarySubstabce
        WITH d, row
        MATCH (c:Case {primaryid: toInteger(row.primaryid)})
        FOREACH (_ IN CASE WHEN row.role = "Primary Suspect" THEN [1] ELSE [] END |
        MERGE (c)-[relate:IS_PRIMARY_SUSPECT { drugSequence: row.drugSequence, route: row.route, doseAmount: row.doseAmount, doseUnit: row.doseUnit, indication: row.indication  }]->(d)
        )
        FOREACH (_ IN CASE WHEN row.role = "Secondary Suspect" THEN [1] ELSE [] END |
        MERGE (c)-[relate:IS_SECONDARY_SUSPECT { drugSequence: row.drugSequence, route: row.route, doseAmount: row.doseAmount, doseUnit: row.doseUnit, indication: row.indication  }]->(d)
        )
        FOREACH (_ IN CASE WHEN row.role = "Concomitant" THEN [1] ELSE [] END |
        MERGE (c)-[relate:IS_CONCOMITANT { drugSequence: row.drugSequence, route: row.route, doseAmount: row.doseAmount, doseUnit: row.doseUnit, indication: row.indication  }]->(d)
        )
        FOREACH (_ IN CASE WHEN row.role = "Interacting" THEN [1] ELSE [] END |
        MERGE (c)-[relate:IS_INTERACTING { drugSequence: row.drugSequence, route: row.route, doseAmount: row.doseAmount, doseUnit: row.doseUnit, indication: row.indication  }]->(d)
        );
        """,
        
        """
        LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/therapy.csv" AS row
        MERGE (t:Therapy { primaryid: toInteger(row.primaryid) })
        WITH t, row
        MATCH (c:Case {primaryid: toInteger(row.primaryid)})
        MERGE (c)-[:RECEIVED]->(t)
        WITH c, t, row
        MATCH (d:Drug { name: row.drugName })
        MERGE (t)-[:PRESCRIBED { drugSequence: row.drugSequence, startYear: coalesce(row.startYear, 1900), endYear: coalesce(row.endYear, 2021) } ]->(d);
        """
    ]

    for query in import_queries:
        try:
            run_query(query)
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Problematic query: {query}")

if __name__ == "__main__":
    import_data()