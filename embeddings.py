import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pickle as pkl
#tokenizer  = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")


'''
Esta clase se utiliza para generar los embeddings de las entidades que se emplean para formar
los grafos, concretamente para tener la estructura de grafos adecuada para la GNN.
'''

class Embeddings():
    def __init__(self, tokenizer, bert_model, device, folder):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.folder = folder

    def run(self):
        device = torch.device('cuda')
        self.bert_model.to(device)
        self.bert_model.eval()

        with open(f"{self.folder}/vocab.pkl", 'rb') as f:
            names = pkl.load(f)

        embs = []
        tensors = self.tokenizer(names, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            for i, j in enumerate(tqdm(names)):
                outputs = self.bert_model(input_ids=tensors["input_ids"][i:i+1].to(device), 
                                    attention_mask=tensors['attention_mask'][i:i+1].to(device))
                out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
                embs.append(out)
        embs = np.concatenate(embs)
        np.save(f"{self.folder}/ent_emb.npy", embs)