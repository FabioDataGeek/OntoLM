import os
import pickle as pkl

'''
Con este código generamos un dict para cada texto donde cada palabra a desambiguar aparece con 
sus etiquetas correctas, con esto cada palabra solo se desambigua una vez, aunque aparezca dos 
veces.
'''

folder_path = '/usrvol/ISABIAL/Anotaciones_v12_2_ISABIAL'

entries = os.listdir(folder_path)
annotations_list = []
for file_path in range(1000):
    print(f"file number: {file_path}" )
    if f"0text{file_path}.ann" in entries:
        file = f"0text{file_path}.ann"
        with open(f"/usrvol/ISABIAL/Anotaciones_v12_2_ISABIAL/{file}", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split('\t')
                id = line[0]
                category_position = line[1]
                category_position = category_position.split(' ')
                category = category_position[0]
                initial_pos = category_position[1]
                final_pos = category_position[2]
                word = line[2]
                if not category in annotations_list:
                    annotations_list.append(category)
                
print(len(annotations_list))
            




