import pickle as pkl
import json
import random
import math

'''
Ejecutamos este script antes de formar los tensores del grafo
'''

with open('/usrvol/data/categorical/graph_statements/graph_statements.pickle', 'rb') as f:
    graphs =  pkl.load(f)

with open('/usrvol/data/categorical/statements.pickle', 'rb') as f:
    statements = pkl.load(f)

assert len(graphs) == len(statements)

size = len(statements)

random_list = list(range(size))
random.shuffle(random_list)

train_size = math.floor(size * 0.8)
dev_size = math.floor(size * 0.1)
test_size = math.floor(size * 0.1)

train_index = random_list[0:train_size]
dev_index = random_list[train_size:train_size+dev_size]
test_index = random_list[train_size+dev_size:]

# Hacemos los archivos train, test y dev para el texto
counter = 0
for i in (train_index):
    with open(f"/usrvol/data/ISABIAL/categorical/statements/train.statements.jsonl", 'a') as fout:
        stm = statements[i]
        stm['id'] = f"train-{counter:05d}"
        print(json.dumps(stm), file=fout)
    counter +=1 

counter = 0
for i in (dev_index):
    with open(f"/usrvol/data/ISABIAL/categorical/statements/dev.statements.jsonl", 'a') as fout:
        stm = statements[i]
        stm['id'] = f"dev-{counter:05d}"
        print(json.dumps(stm), file=fout)
    counter +=1     

counter = 0
for i in (test_index):
    with open(f"/usrvol/data/ISABIAL/categorical/statements/test.statements.jsonl", 'a') as fout:
        stm = statements[i]
        stm['id'] = f"test-{counter:05d}"
        print(json.dumps(stm), file=fout)
    counter +=1   



# Hacemos los archivos train, test y dev para los grafos
graph_list = []
for i in (train_index):
    graph_list.append(graphs[i])
with open('/usrvol/data/ISABIAL/categorical/graphs/train.pickle', 'wb') as f:
    pkl.dump(graph_list, f)


graph_list = []
for i in (dev_index):
    graph_list.append(graphs[i])
with open('/usrvol/data/ISABIAL/categorical/graphs/dev.pickle', 'wb') as f:
    pkl.dump(graph_list, f)


graph_list = []
for i in (test_index):
    graph_list.append(graphs[i])
with open('/usrvol/data/ISABIAL/categorical/graphs/test.pickle', 'wb') as f:
    pkl.dump(graph_list, f)
