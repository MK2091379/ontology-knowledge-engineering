# Gold Price Economic Ontology & Knowledge Graph

![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![Neo4j](https://img.shields.io/badge/Database-Neo4j-00a393.svg)
![Ontology](https://img.shields.io/badge/Knowledge_Graph-OWL%20%7C%20RDF-ff5252.svg)
![Cypher](https://img.shields.io/badge/Query-Cypher-412991.svg)

## 📖 Overview
This Knowledge Engineering project models the complex economic, political, and geopolitical factors influencing global gold prices, specifically focusing on the timeline from the Russia-Ukraine war to the present. Using **Neo4j** and **Arrows**, the project establishes a comprehensive ontology graph to visualize and logically query how variables like oil prices, inflation, market risks, and international sanctions interconnect to impact the economy. 

## ✨ Key Features
*   **Rich Knowledge Graph:** A carefully structured ontology comprising 48 nodes (Concepts and Instances) and 64 relationships (e.g., `positivelyCorrelatedWith`, `caused by`, `sanctionedBy`).
*   **Competency Questions Resolution:** Evaluates complex real-world queries, tracing paths to understand the direct and indirect impacts of the US 2024 Election, Asian economic growth, and Middle East conflicts on gold prices.
*   **Automated Cypher Generation:** Includes a custom Python script (`json_to_cypher.py`) that parses Neo4j Arrows JSON exports to programmatically generate executable `MERGE` Cypher queries.
*   **RDF/OWL Conversion:** Features `csv_to_owl.py`, leveraging the `rdflib` library to convert JSON and CSV graph data into standard OWL formats (`csv_ontology.owl`, `json_ontology.owl`) to ensure Semantic Web compatibility.

## 🏗️ Repository Structure
The repository is organized to separate documentation, visual assets, and execution scripts:
*   `graph_picture_in_svg.svg`: High-resolution visual representation of the ontology network.
*   `codes/neo4j-graph.json` & `data.csv`: Raw exported graph data structures.
*   `codes/output.cypher`: The generated Cypher script ready for immediate Neo4j deployment.
*   `codes/hwii.dump`: Native Neo4j database dump file for quick database restoration.

## 📊 Usage & Deployment
To deploy this knowledge graph into your local Neo4j instance:
1.  **Direct Dump Import:** Use the Neo4j admin tool to load the provided dump file via your terminal: `bin/neo4j-admin load --from=hwii.dump --database=neo4j --force`.
2.  **Cypher Execution:** Alternatively, open `codes/output.cypher` and execute the queries directly in the Neo4j Browser.
3.  **Generate OWL Models:** Run `python csv_to_owl.py` to recreate the RDF/XML ontology files using the provided data.