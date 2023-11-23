import pickle as pkl
import os
from tqdm import tqdm
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

text_path = '/usrvol/experiments/ISABIAL/Anotaciones_v12_2_ISABIAL'
graph_path = "/usrvol/experiments/data/categorical_gran_escala/graphs"
entries = os.listdir(text_path)


with open('/usrvol/experiments/mappings/classes.pickle', 'rb') as f:
    classes = pkl.load(f)

with open('/usrvol/experiments/data/categorical_gran_escala/final_labels.pickle', 'rb') as f:
    final_labels = pkl.load(f)
statements = []
graphs_statements = []

iterator = 0
# iteramos de igual forma que en graph_labels para que coincidan los textos con las etiquetas y los grafos
for file_path in tqdm(entries):
    if file_path.endswith('.txt'):
        file_name = file_path.split('.')[0]
        if file_name.startswith('0') or file_name.startswith('5'):
            text_file = f"{text_path}/{file_path}"
            # obtenemos el texto si es el archivo objetivo
            with open(text_file, 'r') as f:
                text = f.readlines()
                text = text[0].strip()
            # grafos obtenidos de dicho texto
            graphs_file = f"{graph_path}/final_graphs{iterator}.pickle"
            with open(graphs_file, 'rb') as f:
                graphs = pkl.load(f)

            entity_list = []
            for graph in range(len(graphs)):
                entity_list.append(str(list(graphs[graph].keys())[0]))

            # iterate over each term in the final_labels[iterator]
            for term in range(len(final_labels[iterator].copy())):
                statement = {}
                statement["stem"] = text
                objective = entity_list[term]
                statement["objective"] = objective
                categories = final_labels[iterator][term]
                statement['category'] = categories
                statements.append(statement)
                graphs_statements.append(list(graphs[term].values())[0])

            iterator += 1


with open("/usrvol/experiments/data/categorical_gran_escala/statements.pickle", 'wb') as f:
    pkl.dump(statements, f)

with open("/usrvol/experiments/data/categorical_gran_escala/graph_statements/graph_statements.pickle", 'wb') as f:
    pkl.dump(graphs_statements, f)

print("")