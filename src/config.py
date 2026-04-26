#arquivo com informações globais
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#classes da classificação multilabel
class_cols = ['aggressive', 'hate', 'ageism', 'aporophobia', 'body_shame', 'capacitism', 'lgbtphobia', 'political', 'racism', 'religious_intolerance', 'misogyny', 'xenophobia', 'other']

models = {
    "BERTimbau-base": "neuralmind/bert-base-portuguese-cased",
    "BERTimbau-large": "neuralmind/bert-large-portuguese-cased",
    "Albertina": "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac", #talvez trocar para versão menor
    "Bernice": "jhu-clsp/bernice",
    "Gemma": "google/gemma-4-E4B-it",
    "Llama": "meta-llama/Llama-3.2-1B"
}