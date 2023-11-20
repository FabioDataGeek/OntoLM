from embeddings import *


folder_in = '/usrvol/mappings/binary/vocab.txt'
folder_out = '/usrvol/data/binary/embs/ent_emb.npy'
emb = Embeddings(tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
                 bert_model=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),
                 device='cuda:0', folder_in=folder_in, folder_out=folder_out)

emb.run()