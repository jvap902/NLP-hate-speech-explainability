import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src import config

def plotAttributions(df, input_index, save_path=None, show=True):
    # 1. Converter de Wide (uma coluna por classe) para Long (formato do Seaborn)
    # Isso coloca todas as atribuições em uma única coluna 'Score' e os nomes das classes em 'Classe'
    df_long = df.melt(id_vars=["token"], var_name="Class", value_name="Attribution")

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # 2. Plotar usando Seaborn
    # x="token" garante que o eixo X não se repita
    # y="Attribution" é o eixo Y (o valor numérico)
    # hue="Class" cria uma linha colorida para cada coluna original de classe
    plot = sns.lineplot(data=df_long, x="token", y="Attribution", hue="Class", marker='o', linewidth=1.5, palette="husl")
    
    # 3. Ajustes de escala e estética
    y_min, y_max = df_long["Attribution"].min(), df_long["Attribution"].max()
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Evita que o yticks quebre se os valores forem muito pequenos
    if abs(y_max - y_min) > 0.1:
        plt.yticks(np.arange(np.floor(y_min*2)/2, np.ceil(y_max*2)/2 + 0.5, 0.5))

    plt.xticks(rotation=45, ha='right')
    plt.title(f'Análise de Importância por Classe - Input #{input_index}', fontsize=14)
    plt.ylabel('Atribuição (Integrated Gradients)')
    plt.xlabel('Tokens')
    
    # Coloca a legenda para fora para não tampar as linhas
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Labels TuPyE")
    
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path, dpi=300)
    if show: plt.show() 
    plt.close()
    
    
def plotLosses(df: pd.DataFrame, save_path=None, show=True):

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    ax = sns.lineplot(
        data=df, 
        x="Iteration", 
        y="Loss", color="#2c3e50", linewidth=1.5)

    ax.set_title('Model Convergence: Training Loss over Batches', fontsize=14, pad=15)
    ax.set_xlabel('Iterations (Batches)', fontsize=12)
    ax.set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)

    plt.tight_layout()

    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show: plt.show()
    
    plt.close()