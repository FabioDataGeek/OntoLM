import re
import pandas as pd
import glob
import os
import json
import itertools
import glob
import json
import spacy
import scispacy
from scispacy.linking import EntityLinker
from typing import Dict, Tuple
from spacy.language import Language
import pickle as pkl

class Ner():
    
    def __init__(self, spacy_model: str, pipeline_linker: str, resolve_abbreviations: bool, 
                 ontology: str, threshold: str, max_entities_per_mention: str):
        
        self.spacy_model = spacy_model
        self.resolve_abbreviations = resolve_abbreviations
        self.ontology = ontology
        self.threshold = threshold
        self.max_entities_per_mention = max_entities_per_mention
        self.pipeline_linker = pipeline_linker
        self.nlp = spacy.load(self.spacy_model)
        self.nlp.add_pipe(self.pipeline_linker, 
                          config={"resolve_abbreviations": self.resolve_abbreviations, 
                                "linker_name":self.ontology, "threshold":self.threshold, 
                                "max_entities_per_mention": self.max_entities_per_mention})

    def detected_entities(self, text: str):
        doc = self.nlp(text)
        return doc

    def linked_entities(self, doc):
        list_linked_entities = []
        for ent in doc.ents:
            linked_entities = ent._.kb_ents 
            if len(linked_entities) > 0:
                list_linked_entities.append([str(ent), linked_entities])
        return list_linked_entities
        


ner = Ner(spacy_model='en_core_sci_sm', 
          pipeline_linker='scispacy_linker', 
          resolve_abbreviations=True, 
          ontology="umls", 
          threshold="0.80", 
          max_entities_per_mention='100')


#with open('/usrvol/list_linked_entities', 'rb') as f:
#    list_linked_entities = pkl.load(f)


# cargamos el linker para hacer la prueba de los nombres recibidos
linker = EntityLinker(
        resolve_abbreviations=True,
        name="umls",
        threshold=0.80,
        max_entities_per_mention=100)

def connectedEntities(entity: str, linker, list_linked_entities) -> list[list[str], list[str]]:
        entity_list = []
        TUI_list = []

        for entities in range(len(list_linked_entities)):
            target_entity = list_linked_entities[entities][0].lower()
            if entity.lower() == target_entity:
                new_linked_entities = list_linked_entities[entities][1]
                for new_linked_entity in new_linked_entities:
                    new_linked_entity = linker.kb.cui_to_entity[new_linked_entity[0]]
                    new_linked_entity_name =  new_linked_entity.canonical_name
                    if not new_linked_entity_name.lower() == entity.lower():
                        entity_list.append(new_linked_entity.canonical_name)
                        TUI_list.append(new_linked_entity.types)
                break
        return entity_list, TUI_list


def column_of_entities(entity: str, list_linked_entities, linker):
    column_entities = []
    column_TUIs = []
    entities, TUIs = connectedEntities(entity, linker, list_linked_entities)
    column_entities.append(entities)
    column_TUIs.append(TUIs)

    return column_entities[0], column_TUIs[0]


with open('second_entities', 'rb') as f:
    target = pkl.load(f)


def getNextEntitiesAndTUIs(entity, list_linked_entities, linker):
    entities_list = []
    TUIs_list = []
    TUIs = []
    CUIs = list_linked_entities[0][1]
    for CUI in CUIs:
        new_entity = linker.kb.cui_to_entity[CUI[0]]
        new_entity_name = new_entity.canonical_name
        TUI = new_entity.types
        if str(entity) != str(new_entity_name):
            entities_list.append(new_entity_name)
            TUIs_list.append(TUI)

    return entities_list, TUIs_list

def expanded_graph(target, entity_tree, TUI_tree):
    if type(target) is not str:
        for parent_entity in range(len(target)):
            entity_tree.append([])
            TUI_tree.append([])
            expanded_graph(target[parent_entity], 
                           entity_tree[parent_entity], 
                           TUI_tree[parent_entity])
    else:
        doc = ner.detected_entities(target)
        list_linked_entities = ner.linked_entities(doc)
        entities, TUIs = getNextEntitiesAndTUIs(target, list_linked_entities,
                                             linker)
        entity_tree.append(entities)
        TUI_tree.append(TUIs)
        
    return entity_tree, TUI_tree

third_entities, third_TUIs = expanded_graph(target, [], [])

with open('/usrvol/third_entities', 'wb') as f:
    pkl.dump(third_entities, f)

with open('/usrvol/third_TUIs', 'wb') as f:
    pkl.dump(third_TUIs, f)


# Este código funciona bien para todas las conexiones pero 
# anida mas de la cuenta,creando una lista interna más en cada paso