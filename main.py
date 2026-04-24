from src import *

if __name__ == "__main__":
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    for key, value in models.items(): #apenas testando se é possível carregar
        modelc = Model(value, key)
        
        modelc.getLoader(train, test, batch_size=32)
        
        classify.fineTuneModel(modelc, 1)
        
        classify.evaluateModel(modelc, modelc.test_loader)
        
        del modelc