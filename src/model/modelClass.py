from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
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
        self.train_tokenized = self.tokenize(train)
        self.test_tokenized = self.tokenize(test)
        
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
        torch_tokenized_train = self.train_tokenized.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) #coloca no formato do PyTorch
        torch_tokenized_test = self.test_tokenized.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) #coloca no formato do PyTorch
        
        self.train_loader = DataLoader(torch_tokenized_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.data_collator)
        self.test_loader = DataLoader(torch_tokenized_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=self.data_collator)
        
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
        
    def saveModel(self, repeats=0):
        model_data = fileHandler.getJsonInfo(config.fine_tune_info_path, [self.name])[0]
        
        model_data["repeats"] += repeats
        
        fileHandler.updateJson(json_path=config.fine_tune_info_path, fields=[self.name], values=[model_data])
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        
    def tokenize(self, dataset):
        
        tokenizer = self.tokenizer
        
        def tokenizeInstance(instance):
            tokens = tokenizer(instance["text"], truncation=True, max_length=512)
            tokens["labels"] = [[float(instance[c][i]) for c in config.classes] for i in range(len(instance["text"]))]
            return tokens
        
        ds = dataset.map(tokenizeInstance, batched=True, remove_columns=config.classes)
        return ds
    
    def getTrainer(self, epochs_fold, compute_metrics):
        training_args = getTrainingArgs(self, epochs_fold)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tokenized,
            eval_dataset=self.test_tokenized,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )