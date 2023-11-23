import pickle as pkl
import os
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from tqdm import tqdm

# edge_index debe ser una lista de tensores donde cada tensor es de tamaño [2, E]
# edge_type debe ser una lista de tensores donde cada tensor es de tamaño [E]

def GNN_tensors(graph, graph_max_size):
    
    # concepts_ids
    diff = graph_max_size - len(graph)
    p1d = (0, diff)
    concepts_ids = []
    for nodes in graph.nodes:
        concepts_ids.append(nodes)
    concepts_ids = torch.tensor(concepts_ids, dtype=torch.int64)
    concepts_ids = concepts_ids.unsqueeze(0)
    concepts_ids = F.pad(concepts_ids, p1d, "constant", 1)

    # node_type
    node_type = []
    for nodes in graph.nodes(data=True):
        node_type.append(nodes[1]['node_type'])
    node_type = torch.tensor(node_type, dtype=torch.int64)
    node_type = node_type.unsqueeze(0)
    node_type = F.pad(node_type, p1d, "constant", 2)
    
    # Node_scores
    node_scores = torch.zeros(1,graph_max_size,1)

    #Graph size
    graph_size = torch.tensor(len(graph), dtype=torch.int64)

    # edge index
    simple_graph = nx.MultiDiGraph()
    simple_graph.add_nodes_from(graph.nodes())
    simple_graph.add_edges_from(graph.edges())
    data = from_networkx(simple_graph)
    edge_index = data.edge_index

    # edge type
    edge_type = []
    for edge in graph.edges(data=True):
        edge_type.append(edge[2]['rel'])
    edge_type = torch.tensor(edge_type, dtype=torch.int64)
    return concepts_ids, node_type, node_scores, graph_size, edge_index, edge_type

files =['train', 'test', 'dev']
for file in files:
    with open(f"/usrvol/experiments/ISABIAL/categorical_gran_escala/graphs/{file}.pickle", 'rb') as f:
        graph_statements = pkl.load(f)
    graph_statements_tensors = [None] * 6
    for graphs in tqdm(graph_statements):
        for graph in graphs:
            output = GNN_tensors(graph, 200)
            for i in range(len(output)):
                if graph_statements_tensors[i] == None:
                    if i == 3:
                        graph_statements_tensors[i] = output[i].reshape(1)
                    elif i == 4 or i ==5:
                        graph_statements_tensors[i] = [output[i]]
                    else:
                        graph_statements_tensors[i] = output[i]
                else:
                    if i == 3:
                        graph_statements_tensors[i] = torch.cat((graph_statements_tensors[i],output[i].reshape(1)),0)
                    elif i == 4 or i == 5:
                        graph_statements_tensors[i].append(output[i])
                    else:
                        graph_statements_tensors[i] = torch.cat((graph_statements_tensors[i],output[i]),0)
       
    with open(f"/usrvol/experiments/data/categorical_gran_escala/graph_statements_tensors/graph_statements{file}.pickle", 'wb') as f:
        pkl.dump(graph_statements_tensors, f)

print("")