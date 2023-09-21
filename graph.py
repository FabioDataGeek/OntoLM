import networkx as nx
import pickle as pkl

class GraphGenerator():

    def __init__(self) -> None:
        self.graph = nx.Graph()


    '''
    Tenemos creadas las funciones necesarias para conectar el grafo
    tras crear el grafo directo y el inverso, ahora debemos crear los
    métodos necesarios para formar tanto el grafo directo como el 
    inverso
    '''

    def one_way_graph(entities, TUIs, entity_list, TUI_list, graph_index_list, graph_index, reverse: bool):
        for index in range(len(entities)):
            graph_index = graph_index + "." + str(index+1)
            if type(entities[index]) is not str:
                one_way_graph(entities[index], TUIs[index], entity_list, TUI_list, graph_index_list, graph_index, reverse)
            else:
                entity_list.append(entities[index])
                TUI_list.append(TUIs[index])
                if reverse == True:
                    if graph_index[0] != 'R':
                        graph_index = 'R' + graph_index
                    graph_index_list.append(graph_index)
                else:
                    if graph_index[0] != '0':
                        graph_index = '0' + graph_index
                    graph_index_list.append(graph_index)
            previous = graph_index.rindex('.')
            graph_index = graph_index[0:previous]
        return entity_list, TUI_list, graph_index_list
    

    def connection(self, forward:list, backward:list):
        fw_entities, fw_TUIs, fw_indexes = one_way_graph(forward[0], forward[1], 
                                                        forward[2], forward[3], 
                                                        forward[4], forward[5], 
                                                        forward[6])
        
        bw_entities, bw_TUIs, bw_indexes = one_way_graph(backward[0], backward[1], 
                                                        backward[2], backward[3], 
                                                        backward[4], backward[5], 
                                                        backward[6])
        
        for entity in range(len(fw_entities)):
            for bw_entity in range(len(bw_entities)):
                if fw_TUIs[entity] == bw_TUIs[bw_entity]:
                    self.graph.add_edge(fw_indexes[entity], bw_indexes[bw_entity])