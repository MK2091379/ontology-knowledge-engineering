
import json
import csv
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal

EX = Namespace("http://example.org/neo4j-ontology#")


def create_ontology_from_json(json_file, output_owl):

    g = Graph()
    g.bind("ex", EX)

    onto_uri = EX[""]
    g.add((onto_uri, RDF.type, OWL.Ontology))

    nodes = {}

    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "node":
                node_id = data["id"]
                name_str = data["properties"].get("name", "")
                labels = data.get("labels", [])


                node_uri = EX[f"node_{node_id}"]
                nodes[node_id] = node_uri

                if "Concept" in labels:
                    # If it's a concept, treat as owl:Class
                    g.add((node_uri, RDF.type, OWL.Class))
                    # also add a label
                    g.add((node_uri, RDFS.label, Literal(name_str)))
                elif "Instance" in labels:
                    # If it's an instance, treat as owl:NamedIndividual
                    g.add((node_uri, RDF.type, OWL.NamedIndividual))
                    g.add((node_uri, RDFS.label, Literal(name_str)))
                else:
                    # fallback or custom logic
                    g.add((node_uri, RDFS.label, Literal(name_str)))

            elif data["type"] == "relationship":
                rel_label = data["label"]
                start_id = data["start"]["id"]
                end_id = data["end"]["id"]

                start_uri = nodes.get(start_id)
                end_uri = nodes.get(end_id)
                if not start_uri or not end_uri:
                    continue


                if rel_label == "is a":

                    if (start_uri, RDF.type, OWL.Class) in g and (end_uri, RDF.type, OWL.Class) in g:
                        # subClassOf
                        g.add((start_uri, RDFS.subClassOf, end_uri))
                    else:
                        # otherwise, interpret as instance-of
                        g.add((start_uri, RDF.type, end_uri))
                else:
                    # For all other rels, we create an ObjectProperty with the same label
                    prop_uri = EX[rel_label.replace(" ", "_")]
                    # Add an RDF triple:
                    g.add((start_uri, prop_uri, end_uri))

    # Finally, write out RDF/XML
    g.serialize(destination=output_owl, format="xml")
    print(f"Ontology saved to: {output_owl}")


def create_ontology_from_csv(csv_file, output_owl):

    g = Graph()
    g.bind("ex", EX)

    onto_uri = EX[""]
    g.add((onto_uri, RDF.type, OWL.Ontology))

    nodes = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = row.get("type", "")
            if row_type == "node":
                node_id = row["id"]
                name_str = row.get("name", "")
                labels = row.get("labels", "").split("|")

                node_uri = EX[f"node_{node_id}"]
                nodes[node_id] = node_uri
                if "Concept" in labels:
                    g.add((node_uri, RDF.type, OWL.Class))
                    g.add((node_uri, RDFS.label, Literal(name_str)))
                elif "Instance" in labels:
                    g.add((node_uri, RDF.type, OWL.NamedIndividual))
                    g.add((node_uri, RDFS.label, Literal(name_str)))

            elif row_type == "relationship":
                rel_label = row.get("label", "")
                start_id = row.get("start_id", "")
                end_id = row.get("end_id", "")

                start_uri = nodes.get(start_id)
                end_uri = nodes.get(end_id)
                if not (start_uri and end_uri):
                    continue

                if rel_label == "is a":
                    if (start_uri, RDF.type, OWL.Class) in g and (end_uri, RDF.type, OWL.Class) in g:
                        g.add((start_uri, RDFS.subClassOf, end_uri))
                    else:
                        g.add((start_uri, RDF.type, end_uri))
                else:
                    prop_uri = EX[rel_label.replace(" ", "_")]
                    g.add((start_uri, prop_uri, end_uri))

    g.serialize(destination=output_owl, format="xml")
    print(f"Ontology saved to: {output_owl}")


if __name__ == "__main__":
    create_ontology_from_json("neo4j-graph.json", "json_ontology.owl")
    create_ontology_from_csv("data.csv", "csv_ontology.owl")

