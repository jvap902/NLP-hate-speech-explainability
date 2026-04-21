import torch
from src import *

if __name__ == "__main__":
    
    #train, test = loadDataset.loadTuPyE(2000, 2000) #definir tamanhos dos splits
    
    #m1 = Model("google/gemma-4-E4B-it", "gemma-E4B")
    m2 = Model("neuralmind/bert-base-portuguese-cased", "BERTimbau-base")
    m3 = Model("neuralmind/bert-large-portuguese-cased", "BERTimbau-large")