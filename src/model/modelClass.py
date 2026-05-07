from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from .modelUtils import *
from .. import config
from .. import fileHandler

console = Console()

class Model():
    def __init__(self, model_link, model_name):
        self.name = model_name
        self.link = model_link
        
        self.save_dir = f"{config.model_save_dir}/{self.name}"
        
        if Path(self.save_dir).is_dir():
            self.loadModel()
        else:
            self.newModel()
        
        self.model.to(config.device)
        
    def getLoader(self, train, test, batch_size):
        self.train_loader = DataLoader(self.tokenize(train), batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.tokenize(test), batch_size=batch_size, shuffle=False, num_workers=4)
        
    def newModel(self):
        
        console.print(Markdown(f"\n ## Creating new {self.name} \n"))
        
        if 'gemma' in self.name:
            self.model, self.tokenizer = getGemma(self)
        if 'bertimbau' in self.name.lower():
            self.model, self.tokenizer = getBERTimbau(self)
        else:
            raise ValueError("Unsupported model")
        
    def loadModel(self):
        
        console.print(Markdown(f"\n ## Loading saved model at {self.save_dir} \n"))
        
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
        
    def tokenizeInstance(self, instance):
        return self.tokenizer(instance["text"], padding="max_length", truncation=True, max_length=512) #talvez tenha que alterar isso no futuro
        
    def tokenize(self, dataset):
        
        tokenized_dataset = dataset.map(self.tokenizeInstance, batched=True)
        
        return tokenized_dataset.with_format(type="torch", columns=["input_ids", "attention_mask"] + config.classes, device=config.device) #coloca no formato do PyTorch