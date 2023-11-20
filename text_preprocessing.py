import pandas as pd
import os
import glob


folder_path = '/usrvol/ISABIAL/Anotaciones_v2_ISABIAL/txt'

entries = os.listdir(folder_path)
annotations_dict = {}
for file_path in range(1000):
    print(f"file number: {file_path}" )
    if f"text{file_path}.txt" in entries:
        file = f"text{file_path}.txt"
        with open(f"/usrvol/ISABIAL/Anotaciones_v2_ISABIAL/txt/{file}", 'r') as f:
            lines = f.readlines()
            print(lines[0])
    break
    """  for line in lines:
            line = line.strip()
            line = line.split('\t')
            id = line[0]
            category_position = line[1]
            category_position = category_position.split(' ')
            category = category_position[0]
            initial_pos = category_position[1]
            final_pos = category_position[2]
            word = line[2]
            
            print(category, initial_pos, final_pos, word)
    break """


