import json

def generate_cypher(json_file_path, subclass_relation="is a"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    relationships = data['relationships']

    node_dict = {node['id']: node['caption'] for node in nodes}

    parent_nodes = set()
    for rel in relationships:
        if rel['type'].strip().lower() == subclass_relation.strip().lower():
            parent_nodes.add(rel['toId'])

    # Determine node type: Concept (has children) or Instance (no children)
    node_types = {}
    for node_id in node_dict:
        node_types[node_id] = "Concept" if node_id in parent_nodes else "Instance"

    # Generate Cypher Nodes
    cypher_nodes = []
    for node_id, caption in node_dict.items():
        label = node_types[node_id]
        safe_caption = caption.replace("`", "\\`")  # escape backticks
        cypher_nodes.append(f'MERGE (`{safe_caption}`:{label} {{name:"{safe_caption}"}})')

    cypher_rels = []
    for rel in relationships:
        start = node_dict[rel['fromId']].replace("`", "\\`")
        end = node_dict[rel['toId']].replace("`", "\\`")
        rel_type = rel['type'].strip().replace(" ", "_").replace("\t", "").upper()
        cypher_rels.append(f'MERGE (`{start}`)-[:{rel_type}]->(`{end}`)')

    return "\n".join(cypher_nodes + cypher_rels)

json_file = "gold_price (1).json" #Your file path
subclass_relation="is a" #Subclass relation you defined in your arrows graph
cypher_code = generate_cypher(json_file,subclass_relation)

with open('output.cypher', 'w', encoding='utf-8') as f:
    f.write(cypher_code)

print("Cypher script generated successfully as 'output.cypher'")
