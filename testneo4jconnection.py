from neo4j import GraphDatabase
import os
import dotenv

dotenv.load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        for record in result:
            print(record["message"])

test_connection()
