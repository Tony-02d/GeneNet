import sys
from typing import Dict, Any
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd
from sympy.codegen.ast import stderr


def encode_evidence(evidence,encoder,evidence_list):
    if pd.notna(evidence):
        return encoder.transform([[evidence]])[0]
    else:
        return [0] * len(evidence_list)
def build_graph(data):
    evidence_list = data['DirectEvidence'].dropna().unique()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(evidence_list.reshape(-1, 1))

    data['EncodedEvidence'] = data['DirectEvidence'].apply(lambda x: encode_evidence(x, encoder, evidence_list))

    G = nx.DiGraph()

    genes = data[['GeneID', 'GeneSymbol']].drop_duplicates()
    G.add_nodes_from(
        (row.GeneID, {'type': 'gene', 'label': row.GeneSymbol})
        for row in genes.itertuples(index=False)
    )

    diseases = data[['DiseaseID', 'DiseaseName']].drop_duplicates()
    G.add_nodes_from(
        (row.DiseaseID, {'type': 'disease', 'label': row.DiseaseName})
        for row in diseases.itertuples(index=False)
    )

    edges = [
        (row.GeneID, row.DiseaseID, {'evidence': torch.tensor(row.EncodedEvidence, dtype=torch.float)})
        for row in data.itertuples(index=False)
    ]
    G.add_edges_from(edges)

    for node, data in G.nodes(data=True):
        is_gene = 1 if data['type'] == 'gene' else 0
        is_disease = 1 - is_gene



        if is_gene:
            degree = G.out_degree(node)
        else:
            degree = G.in_degree(node)

        G.nodes[node]['x'] = torch.tensor([is_gene, is_disease, degree], dtype=torch.float)

    return G

def get_subgraph_by_label(G: nx.DiGraph, label: str) -> Dict[str, Any]:
    matched_nodes = [n for n, data in G.nodes(data=True) if data.get('label', '').lower() == label.lower()]

    if not matched_nodes:
        return {"error": f"No node found with label '{label}'"}

    node = matched_nodes[0]
    node_type = G.nodes[node].get('type')

    nodes_to_include = [node]

    if node_type == 'gene':

        nodes_to_include.extend(list(G.successors(node)))
    elif node_type == 'disease':

        nodes_to_include.extend(list(G.predecessors(node)))

    subgraph = G.subgraph(nodes_to_include).copy()

    nodes = []
    for n, data in subgraph.nodes(data=True):
        nodes.append({
            "id": str(n),
            "label": data.get('label'),
            "type": data.get('type')
        })

    edges = []
    for u, v, data in subgraph.edges(data=True):
        u_type = subgraph.nodes[u].get('type')
        v_type = subgraph.nodes[v].get('type')


        if u_type == 'gene' and v_type == 'disease':
            edges.append({
                "source": str(u),
                "target": str(v),
                "evidence": data.get('evidence').tolist() if data.get('evidence') is not None else None
            })

    return {
        "nodes": nodes,
        "edges": edges
    }
