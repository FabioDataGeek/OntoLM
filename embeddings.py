import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pickle as pkl
import math
#tokenizer  = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")


'''
Esta clase se utiliza para generar los embeddings de las entidades que se emplean para formar
los grafos, concretamente para tener la estructura de grafos adecuada para la GNN.
'''

class Embeddings():
    def __init__(self, tokenizer, bert_model, device, folder_in, folder_out):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.folder_in = folder_in
        self.folder_out = folder_out
        self.bert_model.to(self.device)


    def run(self):
        self.bert_model.eval()
        with open(f"{self.folder_in}") as f:
            f = f.readlines()
            embs = []
            for i in tqdm(range(len(f))):
                name = [f[i].strip()]
                if len(name[0]) > 512:
                    name = [name[0][:512]]
                tensors = self.tokenizer(name, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.bert_model(input_ids=tensors["input_ids"].to(self.device), 
                                        attention_mask=tensors['attention_mask'].to(self.device))
                    out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
                    embs.append(out)
            embs = np.concatenate(embs)
            np.save(f"{self.folder_out}", embs)










"""                 if final_point + batch <= nlines:
                    initial_point += batch
                    final_point += batch
                else:
                    initial_point += batch
                    diff = nlines-final_point
                    final_point = nlines - diff """