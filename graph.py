import os
import networkx as nx
import pickle as pkl
from utils import character_count

class GraphGenerator():

    def __init__(self, folder):
        self.graph = nx.Graph()
        self.folder = folder

    def connectionMethod1(self):
        '''
        Este método de conexión solamente conecta la última capa de los nodos directos e inversos
        si coincide el TUI de las entidades
        '''
        max_counter = 0
        # Asignamos un conteo para saber cuales son los nodos de la última fila por su número de puntos
        for node in self.graph.nodes:
            counter = character_count(node, '.')
            if counter > max_counter:
                max_counter = counter
        for node in self.graph.nodes:
            if node.startswith('0'):
                # Nodos directos de la última fila
                if character_count(node, '.') == max_counter:
                    for Rnode in self.graph.nodes:
                        if Rnode.startswith('R'):
                            # Nodos inversos de la última fila
                            if character_count(Rnode, '.') == max_counter:
                                TUIs = self.graph.nodes[node]['attr'][1]
                                RTUIs = self.graph.nodes[Rnode]['attr'][1]
                                for TUI in TUIs:
                                    if TUI in RTUIs:
                                        self.graph.add_edge(node, Rnode)


    def connectionMethod2(self):
        '''
        Este método de conexión conecta los nodos directos con los nodos inversos si 
        coinciden en su TUI
        '''
        for node in self.graph.nodes:
            # Comprobación de que se trata de un nodo directo
            if node.startswith('0.'):
                for Rnode in self.graph.nodes:
                    # Comprobación de que se trata de un nodo inverso
                    if Rnode.startswith('R.'):
                        TUIs = self.graph.nodes[node]['attr'][1]
                        RTUIs = self.graph.nodes[Rnode]['attr'][1]
                        for TUI in TUIs:
                            if TUI in RTUIs:
                                self.graph.add_edge(node, Rnode)


    def connectionMethod3(self):
        '''
        Este método de conexión conecta cada nodo del grafo con los nodos que tengan
        el mismo TUI
        '''
        for node in self.graph.nodes:
            # Nos aseguramos de que no toma ni el nodo inicial ni los posibles nodos finales
            if node != '0':
                if not node.startswith('T'):
                    TUIs = self.graph.nodes[node]['attr'][1]
                    for Rnode in self.graph.nodes:
                        # Nos aseguramos de que no se compara consigo mismo
                        if node != Rnode:
                            # Nos aseguramos que el segundo nodo tampoco es el inicial ni los finales
                            if Rnode != '0':
                                if not Rnode.startswith('T'):
                                    RTUIs = self.graph.nodes[Rnode]['attr'][1]
                                    for TUI in TUIs:
                                        if TUI in RTUIs:
                                            self.graph.add_edge(node, Rnode)

    def graph_connection(self, entities, TUIs, node):
        for leaf in range(len(entities)):
            father_node = node
            if type(entities[leaf]) is not str:
                node = node + "." + str(leaf+1)
                self.graph_connection(entities[leaf], TUIs[leaf], node)
            else:
                node = node + "." + str(leaf+1)
                father_index = node.rindex('.')
                father_node = node[0:father_index]
                self.graph.add_node(node, attr= [entities[leaf], TUIs[leaf]])
                self.graph.add_edge(father_node, node)
                node = father_node
            node = father_node
    

    def graph_generator(self, n_hopes, candidate_dict):
        graph = self.graph
        # nodos directos
        node = '0'
        graph.add_node(node, attr= [candidate_dict['candidate'], candidate_dict['TUIs']])
        # nodos inversos
        Rnode = 'R'
        graph.add_node(Rnode)
        for hop in range(n_hopes):
            #nodos directos
            entities = candidate_dict['hop ' + str(hop+1)][0]
            TUIs = candidate_dict['hop ' + str(hop+1)][1]
            self.graph_connection(entities, TUIs, node)
            
            # nodos inversos
            entities = candidate_dict['Rhop ' + str(hop+1)][0]
            TUIs = candidate_dict['Rhop ' + str(hop+1)][1]
            self.graph_connection(entities, TUIs, Rnode)
        
        # Eliminamos el nodo R que no vale pa nah
        self.graph.remove_node('R')

        # Añadimos los nodos de TUIs y las conexiones finales de los mismos
        TUI_list = candidate_dict['TUIs']    
        for TUI in TUI_list:
            self.graph.add_node(TUI)
            for node in self.graph.nodes:
                if node.startswith('R'):
                    if node.rindex('.') == 1:
                        if TUI in self.graph.nodes[node]['attr'][1]:
                            self.graph.add_edge(TUI, node)


        # Conectamos el grafo con el método de conexión seleccionado
        self.connectionMethod1()

    def graph_list(self, n_hopes, candidates_dict):
        for candidate_dict in range(len(candidates_dict)):
            self.graph_generator(n_hopes, candidates_dict[candidate_dict])
            pkl.dump(self.graph, open(f"{self.folder}/graphs/{str(candidate_dict)}.pkl", 'wb'))
            self.graph.clear()

    #TENEMOS QUE MODIFICAR LOS NODOS FINALES, AÑADIENDO COMO ATTR SU RESPECTIVO NOMBRE SEGÚN EL DICCIONARIO QUE LOS MAPEA
    def vocab_generator(self):
        vocab_list = []
        graphs = os.listdir(f"{self.folder}/graphs")
        for graph in graphs:
            g = pkl.load(open(f"{self.folder}/graphs/{graph}", 'rb'))
            for node in g.nodes:
                if node != '0':
                    if not node.startswith('T'):
                        name = g.nodes[node]['attr'][0]
                        if name not in vocab_list:
                            vocab_list.append(name)

        with open(f'{self.folder}/vocab.pkl', 'wb') as f:
            pkl.dump(vocab_list, f)
        return vocab_list
