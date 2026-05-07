from .. import config
from transformers import GemmaForSequenceClassification, AutoTokenizer, BertForSequenceClassification

def getGemma(modelc):
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = GemmaForSequenceClassification.from_pretrained(
        modelc.link,
        num_labels=len(config.classes),
        problem_type="multi_label_classification"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def getBERTimbau(modelc):
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    
    model = BertForSequenceClassification.from_pretrained(
        modelc.link,
        num_labels=len(config.classes),
        problem_type="multi_label_classification"
    )
    
    return model, tokenizer