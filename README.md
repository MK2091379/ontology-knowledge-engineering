# Ontology and Knowledge Engineering

## Description
This repository contains a computational knowledge engineering system and semantic network query engine developed for Knowledge Representation and Reasoning (KRR) over structured lexical ontologies. Utilizing the FarsNet knowledge base (the Persian WordNet ontology), the project implements relational fact extraction, semantic graph construction, hierarchical taxonomy traversal, and automated inference algorithms for relational reasoning (e.g., hypernymy, hyponymy, meronymy, and domain-specific semantic relations).

## Technologies Used
- Python
- NetworkX (Graph Modeling & Network Traversal Algorithms)
- Pandas & NumPy (Relational Data Ingestion & Array Operations)
- Knowledge Graphs & Ontological Engineering
- Semantic Querying & Graph Search
- Lexical Semantic Networks (FarsNet Knowledge Base)

## Repository Structure
- `farsnet_facts.tsv`: Structured knowledge base dataset containing relational triples and ontological facts formatted as `(Head Entity, Relation/Predicate, Tail Entity)` derived from the FarsNet lexical database.
- `main.py`: Core knowledge processing script managing tabular fact parsing, directed semantic graph instantiation, ontology relation querying, semantic path finding, and inference verification.
- `Questions.pdf`: Academic problem specifications outlining the knowledge engineering objectives, query specifications, and ontology modeling tasks.

## Execution

### Prerequisites
Ensure Python 3.8+ is installed on your system.

### Environment Setup
Create and activate an isolated Python virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install pandas numpy networkx matplotlib