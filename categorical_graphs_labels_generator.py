import pickle as pkl
from spacy_utils import *
from entities import *
from utils import *
from graph import *
from embeddings import *
import networkx as nx
from tqdm import tqdm

desambiguation_entities_number = 6000000
classes_number = 7000000
NamesToInt = {}
maximum_hopes = 3

def expanded_graph(target, entity_tree, TUI_tree, ner, le):
    if type(target) is not str and type(target) is not int:
        for parent_entity in range(len(target)):
            entity_tree.append([])
            TUI_tree.append([])
            expanded_graph(target[parent_entity], 
                           entity_tree[parent_entity], 
                           TUI_tree[parent_entity],
                           ner, le)
    else:
        doc = ner.detected_entities(str(target))
        list_linked_entities = ner.linked_entities(doc)
        entities, TUIs = le.getNextEntitiesAndTUIs(target, list_linked_entities)
        for entity in entities:
            entity_tree.append(entity)
        for TUI in TUIs:
            TUI_tree.append(TUI)
       
    return entity_tree, TUI_tree

def subgraph_generator(entities, graph, obtained_entities, TUIs, limit= 200, total_limit = 1000):
    obtained_entities = list(obtained_entities)
    entity_list = []
    triple_list = []
    for entity in entities:
        if entity in graph.nodes():
            counter_in = 0
            for triple in graph.in_edges(entity, data=True):
                if triple[0] not in obtained_entities:
                    target_TUIs = graph.nodes[triple[0]]['TUIs']
                    for TUI in target_TUIs:
                        if TUI in TUIs:
                            if counter_in < limit:
                                entity_list.append(triple[0])
                                obtained_entities.append(triple[0])
                                triple_list.append((triple[0], triple[2]['rel'], triple[1]))
                                counter_in +=1
                                break
            counter_out = 0
            for triple in graph.out_edges(entity, data=True):
                if triple[1] not in obtained_entities:
                    target_TUIs = graph.nodes[triple[1]]['TUIs']
                    for TUI in target_TUIs:
                        if TUI in TUIs:
                            if counter_out < limit:
                                entity_list.append(triple[1])
                                obtained_entities.append(triple[1])
                                triple_list.append((triple[0], triple[2]['rel'], triple[1]))
                                counter_out +=1
                                break
    obtained_entities = list(set(obtained_entities))
    entity_list = list(set(entity_list))
    if len(obtained_entities) > total_limit:
        obtained_entities = obtained_entities[:total_limit]
        entity_list = entity_list[:total_limit]
        new_triple_list = []
        for triple in triple_list:
            if triple[0] in obtained_entities:
                if triple[1] in obtained_entities:
                    new_triple_list.append(triple)

        triple_list = new_triple_list
    return entity_list, triple_list, obtained_entities


with open('/usrvol/mappings/CUIsToInt.pickle', 'rb') as f:
    CuisToInt = pkl.load(f)

with open('/usrvol/mappings/graph.pickle', 'rb') as f:
    G = pkl.load(f)    

with open('/usrvol/mappings/NamestoCUIFinal.pickle', 'rb') as f:
    NamesToCUIs = pkl.load(f)

with open('/usrvol/mappings/IntToCUIs.pickle', 'rb') as f:
    IntToCUIs = pkl.load(f)

with open('/usrvol/mappings/CUItoPreferredNamesFinal.pickle', 'rb') as f:
    CUIToNames = pkl.load(f)

with open('/usrvol/mappings/IntToNames.pickle', 'rb') as f:
    IntToNames = pkl.load(f)

with open('/usrvol/mappings/cuiToTUIs.pickle', 'rb') as f:
    CUItoTUIs = pkl.load(f)

with open('/usrvol/mappings/TUIsToInt.pickle', 'rb') as f:
    TUIToInt = pkl.load(f)

with open('/usrvol/mappings/RelationtoIdentifierTotal.pickle', 'rb') as f:
    RelationToIdentifier = pkl.load(f)

with open('/usrvol/mappings/mapping_definitivo.pickle', 'rb') as f:
    labelMapping = pkl.load(f)

with open('/usrvol/mappings/TUItoSemantic.pickle', 'rb') as f:
    TUItoText = pkl.load(f)

with open('/usrvol/mappings/classes.pickle', 'rb') as f:
    classes = pkl.load(f)

with open('/usrvol/mappings/classesToTextDict.pickle', 'rb') as f:
    classesToText = pkl.load(f)

mappers = (CuisToInt, IntToCUIs, NamesToCUIs, CUIToNames)

# función recursiva que nos expande el grafo a partir de un target devolviendo un diccionario
# con sus conexiones
ner = Ner(spacy_model='en_core_sci_lg', 
          pipeline_linker='scispacy_linker', 
          resolve_abbreviations=True, 
          ontology="umls", 
          threshold="0.85", 
          max_entities_per_mention='100')

le = Linked_Entities(resolve_abbreviations=True, ontology="umls", threshold=0.85, max_entities_per_mention=100)

text_path = '/usrvol/ISABIAL/Anotaciones_v2_ISABIAL/txt'
entries = os.listdir(text_path)
annotations_dict = {}
vocab_mapper = {}
iterator = 0
counter_mapper = 0

for class_ in classes:
    IntToNames[classes_number] = classesToText[class_]
    NamesToInt[class_] = classes_number
    vocab_mapper[classes_number] = counter_mapper
    classes_number += 1
    counter_mapper += 1

for file_path in tqdm(range(1000)):
    iterator += 1
    print(f"file number: {file_path}" )
    if f"text{file_path}.txt" in entries:
        file = f"text{file_path}.txt"
        with open(f"/usrvol/ISABIAL/Anotaciones_v2_ISABIAL/txt/{file}", 'r') as f:
            lines = f.readlines()
        text = lines[0]
        candidates_dict = {}
        doc = ner.detected_entities(text)
        list_linked_entities = ner.linked_entities(doc)
        candidates, candidates_TUIs = le.candidatesChecker(list_linked_entities)
        
        # Con esto evitamos repetir candidatos si aparecen varias veces en el texto
        new_candidates = []
        for candidate in candidates:
            if candidate not in new_candidates:
                new_candidates.append(candidate)
        candidates = new_candidates
        entity_list, TUI_list = le.getEntitiesAndTUIs(list_linked_entities)
        final_graphs = {}
        labels = {}

        # Creamos una lista de Nombres de entidades a desambiguar mapeandolas a un entero
        for candidate in range(len(candidates)):     
            if not candidates[candidate] in NamesToInt.keys():
                IntToNames[desambiguation_entities_number] = candidates[candidate]
                NamesToInt[candidates[candidate]] = desambiguation_entities_number
                vocab_mapper[desambiguation_entities_number] = counter_mapper
                desambiguation_entities_number += 1
                counter_mapper +=1
            else:
                # Si se cumple esta condición quiere decir que el término aparece tanto como candidato como en el vocabulario original
                if NamesToInt[candidates[candidate]] not in vocab_mapper.keys():
                    vocab_mapper[NamesToInt[candidates[candidate]]] = counter_mapper
                    counter_mapper +=1




        # iteramos sobre cada candidato teniendo ya el vocabulario de los candidatos   
        for candidate in range(len(candidates)):
            graph_entities = {'first_entities': [], 'last_entities': [], 'TUIs': []}
            obtained_entities = []
            TUIs = candidates_TUIs[candidate]
            forward_triples_list = []
            backward_triples_list = []
            connected_entities, _ = expanded_graph(candidates[candidate], [], [], ner, le)

            # Nos quedamos solo con las entidades que estén en el vocabulario según la base de datos
            newconnected_entities = []
            for entity in range(len(connected_entities)):
                    if connected_entities[entity] in NamesToCUIs.keys():
                        newconnected_entities.append(CuisToInt[NamesToCUIs[connected_entities[entity].lower()]])
                        
                        # Añadimos estas entidades al vocabulario
                        if not connected_entities[entity] in NamesToInt.keys():
                            IntToNames[desambiguation_entities_number] = connected_entities[entity]
                            NamesToInt[connected_entities[entity]] = desambiguation_entities_number
                            vocab_mapper[desambiguation_entities_number] = counter_mapper
                            desambiguation_entities_number += 1
                            counter_mapper +=1
            
            connected_entities = newconnected_entities
            graph_entities['first_entities'] = connected_entities
  
            # formamos el grafo desde las entidades iniciales
            for hop in range(maximum_hopes):
                connected_entities, triples, obtained_entities = subgraph_generator(connected_entities, G, obtained_entities, TUIs)
                forward_triples_list.extend(triples)
                forward_triples_list = list(set(forward_triples_list))
            Rconnected_entities, RTUIs = le.otherRelatedEntities(candidates[candidate], 
                                                        candidates, 
                                                        candidates_TUIs, 
                                                        entity_list, TUI_list)
            
            # Nos quedamos solo con las entidades que estén en el vocabulario según la base de datos
            newRconnected_entities = []
            for entity in range(len(Rconnected_entities)):
                if Rconnected_entities[entity] in NamesToCUIs.keys():
                    if not Rconnected_entities[entity] in NamesToInt.keys():
                        IntToNames[desambiguation_entities_number] = Rconnected_entities[entity]
                        NamesToInt[Rconnected_entities[entity]] = desambiguation_entities_number
                        vocab_mapper[desambiguation_entities_number] = counter_mapper
                        desambiguation_entities_number += 1
                        counter_mapper +=1

                    if NamesToCUIs[Rconnected_entities[entity].lower()] not in CuisToInt.keys():
                        CuisToInt[NamesToCUIs[Rconnected_entities[entity].lower()]] = counter_mapper
                    newRconnected_entities.append(CuisToInt[NamesToCUIs[Rconnected_entities[entity].lower()]])
            Rconnected_entities = newRconnected_entities
            graph_entities['last_entities'] = Rconnected_entities
            graph_entities['TUIs'] = RTUIs
            statements_graphs = []
           
            # añadimos los TUIs a labels para usarlos como etiquetas
            for TUI in TUIs:
                if not candidates[candidate] in labels.keys():
                    labels[candidates[candidate]] = [labelMapping[TUI]]
                else:
                    labels[candidates[candidate]].append(labelMapping[TUI])

            # Formamos todas las variantes de grafo desde las entidades finales
            for class_ in classes:
                if class_ in labels.keys():
                    graph_last_entities = []

                    # Si la entidad final de la lista de entidades contiene entre sus TUIs el TUI objetivo...
                    for entity in range(len(graph_entities['last_entities'])):
                        if TUI in graph_entities['TUIs'][entity]:
                            # Se añade a graph_last_entities para formar este grafo en particular
                            graph_last_entities.append(graph_entities['last_entities'][entity])
                    if len(graph_last_entities) < 30:
                        final_entities = graph_last_entities
                    else:
                        final_entities = graph_last_entities[:30]
                    # se obtienen las tripletas inversas para este TUI en particular
                    for hop in range(maximum_hopes):      
                        graph_last_entities, triples, obtained_entities = subgraph_generator(graph_last_entities, G, obtained_entities, TUIs)
                        backward_triples_list.extend(triples)
                        backward_triples_list = list(set(backward_triples_list))
                    triples_list = forward_triples_list + backward_triples_list
                    graph = nx.MultiDiGraph()
                    target = candidates[candidate]
                    initial_entities = graph_entities['first_entities']
                    attrs = set()
                    for triple in triples_list:
                        head = triple[0]
                        # Añadimos todas las entidades y relaciones de las tripletas de cada grafo al vocabulario si no existen aún
                        if not IntToNames[head] in NamesToInt.keys():
                            NamesToInt[IntToNames[head]] = head
                            vocab_mapper[head] = counter_mapper
                            counter_mapper +=1
                        elif not head in vocab_mapper.keys():
                            head = NamesToInt[IntToNames[head]]
                        tail = triple[2]
                        # Idem
                        if not IntToNames[tail] in NamesToInt.keys():
                            NamesToInt[IntToNames[tail]] = tail
                            vocab_mapper[tail] = counter_mapper
                            counter_mapper +=1
                        elif not tail in vocab_mapper.keys():
                            tail = NamesToInt[IntToNames[tail]]   
                        graph.add_node(vocab_mapper[head], TUIs='None', node_type=2, node_score=0)
                        graph.add_node(vocab_mapper[tail], TUIs='None', node_type=2, node_score=0)
                        rel = triple[1]
                        weight = 1.
                        graph.add_edge(vocab_mapper[head], vocab_mapper[tail], rel=rel, weight=weight)
                        attrs.add((vocab_mapper[head], vocab_mapper[tail], rel))
                    # añadimos el nodo inicial y sus conexiones
                    graph.add_node(vocab_mapper[NamesToInt[target]], TUIs=TUIs, node_type=0, node_score=0)
                    for entity in initial_entities:
                        if not IntToNames[entity] in NamesToInt.keys():
                            NamesToInt[IntToNames[entity]] = entity
                            vocab_mapper[entity] = counter_mapper
                            counter_mapper +=1
                        elif not entity in vocab_mapper.keys():
                            entity = NamesToInt[IntToNames[entity]]
                        if not entity in graph.nodes:
                            try:
                                T = CUItoTUIs[IntToCUIs[entity]]
                            except:
                                T = 'NOT FOUND'
                        graph.add_node(vocab_mapper[entity], TUIs=[T], node_type=2, node_score=0)
                        graph.add_edge(vocab_mapper[NamesToInt[target]], vocab_mapper[entity], rel=RelationToIdentifier['meaning_of_concept'], weight=1.)
                    # añadimos el nodo final y sus conexiones
                    graph.add_node(vocab_mapper[NamesToInt[class_]], TUIs=['None'], node_type=1, node_score=0)
                    for entity in final_entities:
                        if not IntToNames[entity] in NamesToInt.keys():
                            NamesToInt[IntToNames[entity]] = entity
                            vocab_mapper[entity] = counter_mapper
                            counter_mapper +=1
                        elif not entity in vocab_mapper.keys():
                            entity = NamesToInt[IntToNames[entity]]
                        if not entity in graph.nodes:
                            try:
                                T = CUItoTUIs[IntToCUIs[entity]]
                            except:
                                T = 'NOT FOUND'
                        graph.add_node(vocab_mapper[entity], TUIs=['None'], node_type=2, node_score=0)
                        graph.add_edge(vocab_mapper[entity], vocab_mapper[NamesToInt[class_]], rel=RelationToIdentifier['belongs_to'], weight=1.)
                    statements_graphs.append(graph)
                    backward_triples_list = []
                else:
                    triples_list = forward_triples_list
                    graph = nx.MultiDiGraph()
                    target = candidates[candidate]
                    initial_entities = graph_entities['first_entities']
                    attrs = set()
                    for triple in triples_list:             
                        head = triple[0]
                        # Añadimos todas las entidades y relaciones de las tripletas de cada grafo al vocabulario si no existen aún
                        if not IntToNames[head] in NamesToInt.keys():
                            NamesToInt[IntToNames[head]] = head
                            vocab_mapper[head] = counter_mapper
                            counter_mapper +=1
                        # Esto quiere decir que el término aparece tanto como candidato como en el vocabulario original, teniendo dos números diferentes que vamos a mapear a uno solo
                        elif not head in vocab_mapper.keys():
                            head = NamesToInt[IntToNames[head]]       
                        tail = triple[2]
                        # Idem
                        if not IntToNames[tail] in NamesToInt.keys():
                            NamesToInt[IntToNames[tail]] = tail
                            vocab_mapper[tail] = counter_mapper
                            counter_mapper +=1
                        # Esto quiere decir que el término aparece tanto como candidato como en el vocabulario original, teniendo dos números diferentes que vamos a mapear a uno solo
                        elif not tail in vocab_mapper.keys():
                            tail = NamesToInt[IntToNames[tail]]

                        graph.add_node(vocab_mapper[head], TUIs='none', node_type=2, node_score=0)
                        graph.add_node(vocab_mapper[tail], TUIs='none', node_type=2, node_score=0)
                        rel = triple[1]
                        weight = 1.
                        graph.add_edge(vocab_mapper[head], vocab_mapper[tail], rel=rel, weight=weight)
                        attrs.add((vocab_mapper[head], vocab_mapper[tail], rel))
                    # añadimos el nodo inicial y sus conexiones
                    graph.add_node(vocab_mapper[NamesToInt[target]], TUIs=TUIs, node_type=0, node_score=0)
                    for entity in initial_entities:
                        if not IntToNames[entity] in NamesToInt.keys():
                            NamesToInt[IntToNames[entity]] = entity
                            vocab_mapper[entity] = counter_mapper
                            counter_mapper +=1
                        # Esto quiere decir que el término aparece tanto como candidato como en el vocabulario original, teniendo dos números diferentes que vamos a mapear a uno solo
                        elif not entity in vocab_mapper.keys():
                            entity = NamesToInt[IntToNames[entity]]

                        if not entity in graph.nodes:
                            try:
                                T = CUItoTUIs[IntToCUIs[entity]]
                            except:
                                T = 'NOT FOUND'     
                        graph.add_node(vocab_mapper[entity], TUIs=[T], node_type=2, node_score=0)
                        graph.add_edge(vocab_mapper[NamesToInt[target]], vocab_mapper[entity], rel=RelationToIdentifier['meaning_of_concept'], weight=1.)
                    # añadimos el nodo final y sin conexiones porque no es su clase
                    graph.add_node(vocab_mapper[NamesToInt[class_]], TUIs=['None'], node_type=1, node_score=0)
                    statements_graphs.append(graph)        
            final_graphs[candidates[candidate]] = statements_graphs


        # Debemos eliminar todas las opciones que aparezcan con un NONE en los labels
        candidates_list = []
        for key in labels.keys():
            new_labels = []
            for i in range(len(labels[key])):
                if labels[key][i] != 'NONE':
                    new_labels.append(labels[key][i])
            new_labels = list(set(new_labels))
            if len(new_labels) < 1:
                candidates_list.append(key)
            labels[key] = new_labels

        for key in candidates_list:
            del labels[key]

        with open(f"/usrvol/data/categorical/labels/labels{iterator}.pickle", 'wb') as f:
            pkl.dump(labels, f)
            
        with open(f"/usrvol/data/categorical/graphs/final_graphs{iterator}.pickle", 'wb') as f:
            pkl.dump(final_graphs, f)


# En lugar de usar vocab_dict mapeamos con vocab_mapper usando intToNames
reverse_vocab_mapper = {v: k for k, v in vocab_mapper.items()}
with open("/usrvol/mappings/categorical/vocab.txt", 'w') as f:
    for i in range(len(reverse_vocab_mapper)):
        value = IntToNames[reverse_vocab_mapper[i]]
        print(value, file=f)

with open('/usrvol/mappings/categorical/vocab_mapper.pickle', 'wb') as f:
    pkl.dump(vocab_mapper, f)
print("")