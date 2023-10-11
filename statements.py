import pickle as pkl
from spacy_utils import *
from entities import *
import json
import ast
import os

# Ejemplo de un statement nuevo para WSD

{"id":"train-00000",
 "question": 
 {"stem": "Huntington's disease is a neurodegenerative autosomal disease results due to expansion of polymorphic CAG repeats in the huntingtin gene. Phosphorylation of the translation initiation factor 4E-BP results in the alteration of the translation control leading to unwanted protein synthesis and neuronal function. Consequences of mutant huntington (mhtt) gene transcription are not well known. Variability of age of onset is an important factor of Huntington's disease separating adult and juvenile types. The factors which are taken into account are genetic modifiers, maternal protection i.e excessive paternal transmission, superior ageing genes and environmental threshold. A major focus has been given to the molecular pathogenesis which includes motor disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis part has also been taken care of. This includes genetic testing and both primary and secondary symptoms. The present review also focuses on the genetics and pathology of Huntington's disease.",
    "objective": "results",
    "choices": [{"label": "A", "text": "Laboratory or test results"},
                {"label": "B", "text": "Research activity"},
                {"label": "C", "text": "Intellectual Product"}]},
"answerkey": "A",
"statements":
    [{"statement": "Huntington's disease is a neurodegenerative autosomal disease results due to expansion of polymorphic CAG repeats in the huntingtin gene. Phosphorylation of the translation initiation factor 4E-BP results in the alteration of the translation control leading to unwanted protein synthesis and neuronal function. Consequences of mutant huntington (mhtt) gene transcription are not well known. Variability of age of onset is an important factor of Huntington's disease separating adult and juvenile types. The factors which are taken into account are genetic modifiers, maternal protection i.e excessive paternal transmission, superior ageing genes and environmental threshold. A major focus has been given to the molecular pathogenesis which includes motor disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis part has also been taken care of. This includes genetic testing and both primary and secondary symptoms. The present review also focuses on the genetics and pathology of Huntington's disease. results. Laboratory or test results"}, 
        {"statement": "Huntington's disease is a neurodegenerative autosomal disease results due to expansion of polymorphic CAG repeats in the huntingtin gene. Phosphorylation of the translation initiation factor 4E-BP results in the alteration of the translation control leading to unwanted protein synthesis and neuronal function. Consequences of mutant huntington (mhtt) gene transcription are not well known. Variability of age of onset is an important factor of Huntington's disease separating adult and juvenile types. The factors which are taken into account are genetic modifiers, maternal protection i.e excessive paternal transmission, superior ageing genes and environmental threshold. A major focus has been given to the molecular pathogenesis which includes motor disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis part has also been taken care of. This includes genetic testing and both primary and secondary symptoms. The present review also focuses on the genetics and pathology of Huntington's disease. results. Research Activity"}, 
        {"statement": "Huntington's disease is a neurodegenerative autosomal disease results due to expansion of polymorphic CAG repeats in the huntingtin gene. Phosphorylation of the translation initiation factor 4E-BP results in the alteration of the translation control leading to unwanted protein synthesis and neuronal function. Consequences of mutant huntington (mhtt) gene transcription are not well known. Variability of age of onset is an important factor of Huntington's disease separating adult and juvenile types. The factors which are taken into account are genetic modifiers, maternal protection i.e excessive paternal transmission, superior ageing genes and environmental threshold. A major focus has been given to the molecular pathogenesis which includes motor disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis part has also been taken care of. This includes genetic testing and both primary and secondary symptoms. The present review also focuses on the genetics and pathology of Huntington's disease. results. Intellectual Product"}]}

with open('TUI_MAPPER.txt', "r") as f:
    data = f.read().splitlines()

TUI_MAPPER = {}

for line in range(len(data)):
    index = data[line].index(",")
    TUI_MAPPER[data[line][0:index]] = data[line][index+1:]
    
del TUI_MAPPER['tui']

class Statements():
    def __init__(self, ner, le, mode, folder):
        self.ner = ner
        self.le = le
        self.mode = mode
        self.folder = folder
        assert self.mode in ["train", "test", "dev"], "mode must be one of train, test, dev"

    def generate_statements(self):
        path = os.listdir(self.folder)
        for file in range(len(path)):
            with open(f"{path[file]}", "rb") as f:
                text = f.read()
            doc = self.ner.detected_entities(text)
            list_linked_entities = self.ner.linked_entities(doc)
            candidates, candidates_TUIs = self.le.candidatesChecker(list_linked_entities)
            statements_list = []
            for candidate in range(len(candidates)):
                statement = {}
                statement["id"] = f"{self.mode}-{file}-{candidate}"
                statement["question"] = {}  
                statement["question"]["stem"] = text
                statement["question"]["objective"] = candidates[candidate]
                statement["question"]["choices"] = []
                statement["statements"] = []
                for TUI in range(len(candidates_TUIs[candidate])):
                    TUI_text = TUI_MAPPER[candidates_TUIs[candidate][TUI]]
                    statement["question"]["choices"].append({"label": str(TUI), "text": TUI_text})
                    statement["statements"].append({"statement": f"{text} {candidates[candidate]}. {TUI_text}."})
                    statement["answerkey"] = 0
                statements_list.append(statement)
        return statements_list