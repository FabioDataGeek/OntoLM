import pickle as pkl
import os

'''
Con este archivo cogemos los grafos que tenemos ya generados en sus respectivos diccionarios
y los modificamos para que aparezcan como valor de una llave, la llave es la categoría
para la que se ha creado dicho grafo.
'''

labels_path = '/usrvol/data/binary/labels'
graphs_path = '/usrvol/data/binary/graphs'

for i in range(1001):
    try:
        with open(f"/usrvol/data/graphs/final_graphs{i}.pickle", 'rb') as f:
            graphs = pkl.load(f)
        with open(f"/usrvol/data/labels/labels{i}.pickle", 'rb') as f:
            labels = pkl.load(f)
    except:
        continue
    
    for term in labels.keys():
        dictionary = {}
        for j in range(len(labels[term])):
            label = labels[term][j]
            dictionary[label] = graphs[term][j]
            graphs[term][j] = {label:graphs[term][j]}
        graphs[term] = dictionary
    with open(f"/usrvol/data/graphs_labels/final_graphs{i}.pickle", 'wb') as f:
        pkl.dump(graphs, f)

print("")

