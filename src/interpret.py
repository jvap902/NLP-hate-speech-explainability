from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from typing import Tuple, List

def explainPrediction(modelc, inputs: List[Tuple[str, int]]):
    modelc.model.eval()
    
    integrated_gradients = IntegratedGradients(modelc.model)
    attr = []
    tokens = []
    
    for text, label_idx in inputs:
        
        t = modelc.tokenizeInstance(text)
    
        a = integrated_gradients.attribute(t, target=label_idx, n_steps=200)
        
        print(a)
    
        attr.append(a)
        tokens.append(t)
    
    return attr, tokens


def plotAttributions(attributions, tokens, label_name):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), attributions, align='center')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel('Attribution Score')
    plt.title(f'Feature Importance for Label: {label_name}')
    plt.show()