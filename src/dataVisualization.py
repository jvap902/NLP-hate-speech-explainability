import seaborn as sns
import matplotlib as plt

def plotAttributions(attributions, tokens, label_name, save_path=None, show=True):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), attributions, align='center')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel('Attribution Score')
    plt.title(f'Feature Importance for Label: {label_name}')
    if save_path != None: plt.savefig(save_path, format='png', dpi=100)
    if show: plt.show()