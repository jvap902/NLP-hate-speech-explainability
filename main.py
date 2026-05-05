from src import *
import torch

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    for key, value in models.items(): #apenas testando se é possível carregar
        modelc = Model(value, key)
        
        modelc.getLoader(train, test, batch_size=32)
        
        classify.evaluateModel(modelc, modelc.test_loader)
        
        classify.fineTuneModel(modelc, 1)
        
        classify.evaluateModel(modelc, modelc.test_loader)
        
        modelc.saveModel()
        
        del modelc