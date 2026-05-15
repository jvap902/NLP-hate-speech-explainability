from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from .modelUtils import *
from .. import config
from .. import fileHandler
from ..logging import LivePanel

class Model():
    def __init__(self, model_link, model_name):
        self.name = model_name
        self.link = model_link
        
        self.save_dir = f"{config.model_save_dir}/{self.name}"
        
        self.panel = LivePanel("Loading Model", color="green")
        self.panel.start()
        
        if Path(self.save_dir).is_dir():
            self.loadModel()
        else:
            self.newModel()
        
        self.panel.addMessage("### Model loaded")
        self.panel.stop()
        
        self.model.to(config.device)
        
    def getLoader(self, train, test, batch_size):
        self.train_dataset = train
        self.test_dataset = test
        
        self.train_loader = DataLoader(self.tokenize(train), batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.tokenize(test), batch_size=batch_size, shuffle=False, num_workers=4)
        
    def newModel(self):
        
        self.panel.addMessage(f"\n ## Creating new {self.name} \n")
        
        if 'gemma' in self.name:
            self.model, self.tokenizer = getGemma(self)
        if 'bertimbau' in self.name.lower():
            self.model, self.tokenizer = getBERTimbau(self)
        else:
            raise ValueError("Unsupported model")
        
    def loadModel(self):
        
        self.panel.addMessage(f"\n ## Loading saved model at {self.save_dir} \n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.save_dir,
            num_labels=len(config.classes),
            problem_type="multi_label_classification"
        )
        
    def saveModel(self, trained_epochs=0):
        fileHandler.updateJson(json_path=config.fine_tune_info_path, fields=[self.name], values=[trained_epochs], increment=[True])
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        
    def tokenize(self, dataset):
        
        tokenizer = self.tokenizer
        
        def tokenizeInstance(instance):
            return tokenizer(instance["text"], padding="max_length", truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenizeInstance, batched=True)
        
        return tokenized_dataset.with_format(type="torch", columns=["input_ids", "attention_mask"] + config.classes) #coloca no formato do PyTorch