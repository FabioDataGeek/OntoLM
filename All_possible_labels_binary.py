import os
import pickle as pkl


'''
Con este archivo tomamos las anotaciones que vamos a utilizar y filtramos las
labels que hemos obtenido con spacy para eliminar aquellas palabras que hemos
detectado pero que no se encuentran entre las etiquetas.

Creamos las etiquetas definitivas para cada palabra detectada del texto si aparece
en los datos etiquetados a mano, de forma (categoria, True) o (categoria, False) y esto
en un diccionario de diccionarios donde las primeras llaves son los identificadores del texto
'''


final_labels = {}
annotation_path = '/usrvol/data/binary/annotations/annotations_dict_filtered.pickle'
labels_obtained_path = '/usrvol/data/binary/labels'

with open(annotation_path, 'rb') as f:
    annotation_dict = pkl.load(f)

for i in annotation_dict.keys():
    final_labels[i] = {}
    with open(f"/usrvol/data/binary/labels/labels{i}.pickle", 'rb') as f:
        labels = pkl.load(f)
    for term in labels:
        if term in annotation_dict[i].keys():
            final_labels[i][term] = []
            for label in labels[term]:
                if label in annotation_dict[i][term]:
                    final_labels[i][term].append((label, True))
                else:
                    final_labels[i][term].append((label, False))

with open('/usrvol/data/binary/final_labels.pickle', 'wb') as f:
    pkl.dump(final_labels, f)
