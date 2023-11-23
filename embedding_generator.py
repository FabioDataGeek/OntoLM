from embeddings import *


folder_in = "/usrvol/experiments/mappings/categorical_gran_escala/vocab.txt"
folder_out = "/usrvol/experiments/data/categorical_gran_escala/embs/ent_emb.npy"
emb = Embeddings(tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
                 bert_model=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
                 device='cuda:0', folder_in=folder_in, folder_out=folder_out)

emb.run()