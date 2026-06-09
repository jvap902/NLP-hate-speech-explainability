import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import config

def plotAttributions(df: pd.DataFrame, save_path=None, show=True):
    # 1. Converter de Wide (uma coluna por classe) para Long (formato do Seaborn)
    # Isso coloca todas as atribuições em uma única coluna 'Score' e os nomes das classes em 'Classe'
    df = df.drop(columns=["text_id"])
    df['token_position'] = np.arange(len(df))
    
    df_long = df.melt(id_vars=["token", "token_position"], var_name="Class", value_name="Attribution")
    
    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")

    # 2. Plotar usando Seaborn
    # x="token" garante que o eixo X não se repita
    # y="Attribution" é o eixo Y (o valor numérico)
    # hue="Class" cria uma linha colorida para cada coluna original de classe
    subtle_dashes = ["", (4, 2)] * 6  # repeats to cover all 13 classes
    
    plot = sns.lineplot(
        data=df_long, 
        x="token_position", 
        y="Attribution", 
        hue="Class", 
        style="Class",
        dashes=subtle_dashes, # Cleans up the line styles
        linewidth=2.0, 
        palette="husl"
    )
    
    # 3. Ajustes de escala e estética
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlim(df['token_position'].min(), df['token_position'].max())
    
    y_min, y_max = df_long["Attribution"].min(), df_long["Attribution"].max()
    
    y_min, y_max = math.floor(y_min / 5) * 5, math.ceil(y_max / 5) * 5 # arredonda valores para multiplo de 5
    
    # Evita que o yticks quebre se os valores forem muito pequenos
    ticks = np.linspace(y_min, y_max, 20)
    ticks = np.round(ticks, 2)
    plt.yticks(ticks)

    plt.xticks(ticks=df['token_position'], labels=df['token'], rotation=45, ha='right')
    plt.title(f'Análise de Importância por Classe', fontsize=14)
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
    
def plotModelComparison(df: pd.DataFrame, target_class: str, model_names: list, save_path=None, show=True):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    palette     = sns.color_palette("tab10", n_colors=len(model_names))
    line_styles = ["-", "--"]

    longest = df.groupby('model')['word'].count().idxmax()
    x_labels = df[df['model'] == longest]['word'].tolist()

    for idx, model_name in enumerate(model_names):
        subset = df[df['model'] == model_name].reset_index(drop=True)
        if subset.empty:
            continue

        x_positions = range(len(subset))

        plt.plot(
            x_positions,
            subset['attribution'],
            label=model_name,
            color=palette[idx],
            linestyle=line_styles[idx % len(line_styles)],
            marker='o',
            linewidth=1.5,
        )
        
    y_min, y_max = df["attribution"].min(), df["attribution"].max()
    
    y_min, y_max = math.floor(y_min / 5) * 5, math.ceil(y_max / 5) * 5 # arredonda valores para multiplo de 5
    
    # Evita que o yticks quebre se os valores forem muito pequenos
    ticks = np.linspace(y_min, y_max, 20)
    ticks = np.round(ticks, 2)
    plt.yticks(ticks)

    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha='right')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Attribution Comparison — {target_class}', fontsize=13)
    plt.ylabel('Attribution (avg over subwords)')
    plt.xlabel('Words')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Model")
    plt.tight_layout()

    if save_path: plt.savefig(save_path, dpi=300)
    if show: plt.show()
    plt.close()