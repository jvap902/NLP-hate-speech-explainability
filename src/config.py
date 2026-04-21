#arquivo com informações globais
from torch.cuda import is_available

device = "cuda" if is_available() else "cpu"

#classes da classificação multilabel
class_cols = ['aggressive', 'hate', 'ageism', 'aporophobia', 'body_shame', 'capacitism', 'lgbtphobia', 'political', 'racism', 'religious_intolerance', 'misogyny', 'xenophobia', 'other']
