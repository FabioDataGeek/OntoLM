import pickle as pkl
import gc
from spacy_utils import *
from entities import *
from utils import *

garbage = []

text = """
Huntington's disease is a neurodegenerative autosomal disease 
results due to expansion of polymorphic CAG repeats in the huntingtin gene. 
Phosphorylation of the translation initiation factor 4E-BP results in the 
alteration of the translation control leading to unwanted protein synthesis 
and neuronal function. Consequences of mutant huntington (mhtt) gene 
transcription are not well known. Variability of age of onset is an important 
factor of Huntington's disease separating adult and juvenile types. The factors 
which are taken into account are genetic modifiers, maternal protection i.e 
excessive paternal transmission, superior ageing genes and environmental threshold. 
A major focus has been given to the molecular pathogenesis which includes motor 
disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis 
part has also been taken care of. This includes genetic testing and both primary 
and secondary symptoms. The present review also focuses on the genetics and 
pathology of Huntington's disease."
"""

ner = Ner(spacy_model='en_core_sci_sm', 
          pipeline_linker='scispacy_linker', 
          resolve_abbreviations=True, 
          ontology="umls", 
          threshold="0.80", 
          max_entities_per_mention='100')

entities = Linked_Entities()

Nhop= 3                                     # Número de saltos que queremos hacer
total_dict = {}

for text in range(len(texts)):             
    candidates_dict = {}
    doc = ner.detected_entities(text)
    list_linked_entities = ner.linked_entities(doc)
    candidates, candidates_TUIs = entities.candidatesChecker(list_linked_entities)
    entity_list, TUI_list = getEntitiesAndTUIs(list_linked_entities)

    for candidate in range(len(candidates)):
        candidates_dict[candidate] = {}
        candidates_dict[candidate]['candidate'] = candidates[candidate]
        candidates_dict[candidate]['TUIs'] = candidates_TUIs[candidate]
        candidates_dict[candidate]['RelatedEntities'] = entities.otherRelatedEntities(candidates[candidate], candidates, 
                                                        candidates_TUIs, list_linked_entities, entity_list, TUI_list)
        
        
        
    

'''
Para obtener la segunda vuelta de entidades tenemos que generar otro list_linked_entities 
a partir de los datos obtenidos del primero.
'''

''' 
Podemos hacerlo de forma recursiva, comprobando que la posición en la que se encuentra
la lista tiene un str en lugar de una lista y en caso de que tenga una lista seguir anidando,
si no tiene una lista deberá llamar a la función column_of_entities sobre el nivel de
profundidad actual
'''



first_entities, first_TUIs = entities.column_of_entities(candidates[candidate], list_linked_entities)

second_entity_list = []
second_TUI_list = []
for parent_entity in range(len(first_entities)):
    doc = ner.detected_entities(first_entities[parent_entity])    # entidades conectadas a Test Results
    list_linked_entities = ner.linked_entities(doc)               # listado de estas entidades según UMLS
    second_entities, second_TUIs = entities.column_of_entities([first_entities[parent_entity], list_linked_entities])
    second_entity_list.append(second_entities)
    second_TUI_list.append(second_TUIs)



third_entity_list = []
third_TUI_list = []
for parent_entity1 in range(len(second_entity_list)):
    third_entity_list.append([])
    third_TUI_list.append([])
    for parent_entity2 in range(len(second_entity_list[parent_entity1])):
        doc = ner.detected_entities(second_entity_list[parent_entity1][parent_entity2])
        list_linked_entities = ner.linked_entities(doc)
        third_entities, third_TUIs = entities.column_of_entities([second_entity_list[parent_entity1][parent_entity2]], list_linked_entities)
        third_entity_list[parent_entity1].append(third_entities)
        third_TUI_list[parent_entity1].append(third_TUIs)


connections = {}
target = candidates[candidate]
for hop in range(len(Nhop)):
    if type(target) is str:
        entities, TUIs = entities.column_of_entities(target, list_linked_entities)
        connections[str(hop+1)+ 'entities'] = entities
        connections[str(hop+1)+ 'TUIs'] = TUIs
    else:
        pass


# función recursiva que nos expande el grafo a partir de un target devolviendo un diccionario
# con sus conexiones

def expanded_graph(self, target, list_linked_entities, entity_tree, TUI_tree):
    if type(target) is str:
        entities, TUIs = entities.column_of_entities(target, list_linked_entities)
        entity_tree = entities
        TUI_tree = TUIs
    else:
        for parent_entity in range(len(target)):
            entity_tree.append([])
            TUI_tree.append([])
            doc = ner.detected_entities(target[parent_entity])    
            list_linked_entities = ner.linked_entities(doc)
            expanded_graph(self, target[parent_entity], 
                           list_linked_entities, entity_tree[parent_entity], 
                           TUI_tree[parent_entity])
    
    return entity_tree, TUI_tree
        
       



total_dict[text] = candidates_dict







garbage.append(ner)
garbage.append(doc)
garbage.append(linked_entities)

'''
Eliminamos el objeto que llama a la pipeline y hacemos un collect con el garbage 
collector de python recuperando así la memoria correspondiente
'''

garbage_out(garbage)