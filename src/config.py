#arquivo com informações globais
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#classes da classificação multilabel
dataset_classes = ['ageism', 'aporophobia', 'body_shame', 'capacitism', 'lgbtphobia', 'political', 'racism', 'religious_intolerance', 'misogyny', 'xenophobia', 'other']

model_classes = ["ageism", "aporophobia", "body_shame", "capacitism", "lgbtphobia", "political", "racism", "religious_intolerance", "misogyny", "xenophobia", "other", "not_hate"]

models = {
    "BERTimbau-base": "Silly-Machine/TuPy-Bert-Base-Multilabel",
    "BERTimbau-large": "Silly-Machine/TuPy-Bert-Large-Multilabel",
    "Albertina": "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac",
    "Bernice": "jhu-clsp/bernice"
    #"Gemma": "google/gemma-4-E4B-it", talvez em um trabalho futuro
    #"Llama": "meta-llama/Llama-3.2-1B"
}

model_save_dir = "fine-tuned-models"

fine_tune_info_path = f"{model_save_dir}/info.json"

ig_dir = "ig-results"

losses_dir = "losses"