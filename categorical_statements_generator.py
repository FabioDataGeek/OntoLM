import pickle as pkl
import os
'''
En este archivo generamos los statements de los grafos y del texto que vamos a pasar
al modelo, es decir, cada instancia de los datos utilizados en el formato que acepta 
el modelo.
'''

# Ejemplo de un statement para resolver el problema con etiquetas categóricas
{
"id":"train-00000",
 "stem": "Huntington's disease is a neurodegenerative autosomal disease results due to expansion of polymorphic CAG repeats in the huntingtin gene. Phosphorylation of the translation initiation factor 4E-BP results in the alteration of the translation control leading to unwanted protein synthesis and neuronal function. Consequences of mutant huntington (mhtt) gene transcription are not well known. Variability of age of onset is an important factor of Huntington's disease separating adult and juvenile types. The factors which are taken into account are genetic modifiers, maternal protection i.e excessive paternal transmission, superior ageing genes and environmental threshold. A major focus has been given to the molecular pathogenesis which includes motor disturbance, cognitive disturbance and neuropsychiatric disturbance. The diagnosis part has also been taken care of. This includes genetic testing and both primary and secondary symptoms. The present review also focuses on the genetics and pathology of Huntington's disease.",
 "objective": "hungtintin gene",
 "category": ["AminoAcidPeptideOrProtein", "GeneOrGenome", "BiologicallyActiveSubstance"],
}

"""
En este caso no hace falta poner label como tal ya que tenemos 40 categorias y la presencia
de la categoría en el statement indica que es verdadera, mientras que sino será falso
"""

folder_path = '/usrvol/ISABIAL/Anotaciones v0 ISABIAL'
entries = os.listdir(folder_path)
with open('/usrvol/data/final_labels.pickle', 'rb') as f:
    final_labels = pkl.load(f)
statements = []
graphs_statements = []
for i in final_labels.keys():
    file = f"text{i-1}.txt"

    with open(f"/usrvol/data/categorical/graphs/final_graphs{i}.pickle", 'rb') as f:
        graphs = pkl.load(f)

    with open(f"/usrvol/ISABIAL/Anotaciones v0 ISABIAL/{file}", 'r') as f:
        text = f.readlines()
        text = text[0].strip()
        for term in final_labels[i].keys():
            statement = {}
            statement["stem"] = text
            statement["objective"] = term
            statement['category'] = []
            for label in final_labels[i][term]:
                if label[0] == "":
                    continue
                else:
                    statement['category'].append(label[0])
            statements.append(statement)
            graphs_statements.append(graphs[term])

with open("/usrvol/data/categorical/statements.pickle", 'wb') as f:
    pkl.dump(statements, f)

with open("/usrvol/data/categorical/graph_statements/graph_statements.pickle", 'wb') as f:
    pkl.dump(graphs_statements, f)

print("")