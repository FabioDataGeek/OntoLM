import os
import pickle as pkl
from tqdm import tqdm

'''
Con este archivo tomamos las anotaciones que vamos a utilizar y filtramos las
labels que hemos obtenido con spacy para eliminar aquellas palabras que hemos
detectado pero que no se encuentran entre las etiquetas.

Creamos las etiquetas definitivas para cada palabra detectada del texto si aparece
en los datos etiquetados a mano, de forma categoria, si no se menciona la categoria se considera falsa
'''


final_labels = {}
labels_obtained_path = '/usrvol/experiments/data/categorical_gran_escala/labels'

entries = os.listdir(labels_obtained_path)
for i in tqdm(range(len(entries))):

    final_labels[i] = []
    with open(f"/usrvol/experiments/data/categorical_gran_escala/labels/labels{i}.pickle", 'rb') as f:
        labels = pkl.load(f)
    for term in range(len(labels)):
            final_labels[i].append([])
            final_labels[i][term] = list(labels[term].values())[0]

with open('/usrvol/experiments/data/categorical_gran_escala/final_labels.pickle', 'wb') as f:
    pkl.dump(final_labels, f)
