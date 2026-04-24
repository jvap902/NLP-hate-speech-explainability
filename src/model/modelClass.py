from torch.utils.data import DataLoader
from .modelUtils import *
from .. import config

class Model():
    def __init__(self, model_link, model_name):
        self.name = model_name
        self.link = model_link
        
        self.getModel()
        
        self.model.to(config.device)
        
    def getLoader(self, train, test, batch_size):
        self.train_loader = DataLoader(self.tokenize(train), batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.tokenize(test), batch_size=batch_size, shuffle=False, num_workers=4)
        
    def getModel(self):
        if 'gemma' in self.name:
            self.model, self.tokenizer = getGemma(self)
        if 'bertimbau' in self.name.lower():
            self.model, self.tokenizer = getBERTimbau(self)
        else:
            raise ValueError("Unsupported model")
        
    def tokenizeInstance(self, instances):
        return self.tokenizer(instances["text"], padding="max_length", truncation=True, max_length=512) #talvez tenha que alterar isso no futuro
        
    def tokenize(self, dataset):
        
        tokenized_dataset = dataset.map(self.tokenizeInstance, batched=True, remove_columns=dataset.column_names)
        
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + self.config.class_cols) #coloca no formato do PyTorch
        
        return tokenized_dataset