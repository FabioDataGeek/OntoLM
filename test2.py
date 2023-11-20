from utils import *
from graph import *
import pickle as pkl


gg = GraphGenerator('/usrvol/GitHub2/results/')


with open('/usrvol/GitHub2/candidates_dict', 'rb') as f:
    candidates_dict = pkl.load(f)

lista = gg.graph_list(3, candidates_dict)


# Implementar el siguiente código mañana en las funciones de graph.py para conectar los nodos inversos con los TUIs.
# seguir escribiendo el código que conectará los grafos directos e inversos
f.remove_node('R')


for node in f.nodes:
    if node.startswith('R'):
        if node.rindex('.') == 1:
            print(f.nodes[node]['attr'][1])


TUI_list = candidates_dict[0]['TUIs']


for TUI in TUI_list:
    f.add_node(TUI)
    for node in f.nodes:
        if node.startswith('R'):
            if node.rindex('.') == 1:
                if TUI in f.nodes[node]['attr'][1]:
                    f.add_edge(TUI, node)