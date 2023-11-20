import os
import pickle as pkl

annotation_path = '/usrvol/data/annotations/annotations_dict.pickle'
labels_obtained_path = '/usrvol/data/binary/labels'

with open(annotation_path, 'rb') as f:
    annotation_dict = pkl.load(f)


entries = os.listdir(labels_obtained_path)


annotation_dict_filtered = {}
for i in annotation_dict.keys():
    annotation_dict_filtered[i] = {}
    with open(f"/usrvol/data/binary/labels/labels{i}.pickle", 'rb') as f:
        labels = pkl.load(f)
    for key in annotation_dict[i].keys():
        if key in labels.keys():
            annotation_dict_filtered[i][key] =  annotation_dict[i][key]  


with open('/usrvol/data/binary/annotations/annotations_dict_filtered.pickle', 'wb') as f:
    pkl.dump(annotation_dict_filtered, f)

print("")