from src import *

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    for key, value in models.items(): #apenas testando se é possível carregar
        modelc = Model(value, key)
        
        modelc.getLoader(train, test, batch_size=32)
        
        classify.evaluateModel(modelc, modelc.test_loader)
        
        modelc, losses = classify.fineTuneModel(modelc, 15, epoch_save=True, colab=False) #colab=True caso esteja rodando no colab
        
        classify.evaluateModel(modelc, modelc.test_loader)
        
        #modelc.saveModel()
        
        attr, tokens = interpret.explainPrediction(modelc, [("eu queria dar um soco nele", 0)])
        dataVisualization.plotAttributions(attr[0], tokens[0], "aggressive", save_path=f"{modelc.name}.png", show=False)
        
        del modelc