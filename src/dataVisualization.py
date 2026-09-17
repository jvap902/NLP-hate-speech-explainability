import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import config

def plotAttributions(df: pd.DataFrame, save_path=None, show=True):

    if df is None or df.empty:
        print(f"⚠️ DataFrame vazio recebido. Plot ignorado.")
        return

    df = df.drop(columns=["text_id"])
    df['token_position'] = np.arange(len(df))
    
    df_long = df.melt(id_vars=["token", "token_position"], var_name="Class", value_name="Attribution")
    
    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")

    subtle_dashes = ["", (4, 2)] * 6  # repeats to cover all 13 classes
    
    plot = sns.lineplot(
        data=df_long, 
        x="token_position", 
        y="Attribution", 
        hue="Class", 
        style="Class",
        dashes=subtle_dashes, # Cleans up the line styles
        linewidth=2.0, 
        palette="husl",
        markers=True
    )
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    if len(df) > 1:
        plt.xlim(df['token_position'].min(), df['token_position'].max())
    else:
        # Se tiver apenas 1 token na frase inteira, centraliza o gráfico
        plt.xlim(-1, 1)
    
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
    
def plotModelComparison(df: pd.DataFrame, save_path=None, show=True):
    
    df['position'] = np.arange(len(df))
    
    df_long = df.melt(id_vars=["word", "position"], var_name="Model", value_name="Attribution")    
    
    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")
        
    plot = sns.lineplot(
        data=df_long, 
        x="position", 
        y="Attribution", 
        hue="Model", 
        style="Model",
        linewidth=2.0,
        dashes=True,
        markers=True,
        palette="husl"
    )
    
    # 3. Ajustes de escala e estética
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlim(df['position'].min(), df['position'].max())
    
    y_min, y_max = df_long["Attribution"].min(), df_long["Attribution"].max()
    
    y_min, y_max = math.floor(y_min / 5) * 5, math.ceil(y_max / 5) * 5 # arredonda valores para multiplo de 5
    
    # Evita que o yticks quebre se os valores forem muito pequenos
    ticks = np.linspace(y_min, y_max, 20)
    ticks = np.round(ticks, 2)
    plt.yticks(ticks)

    plt.xticks(ticks=df['position'], labels=df['word'], rotation=45, ha='right')
    plt.title(f'Comparação de atribuições de modelos', fontsize=14)
    plt.ylabel('Atribuição (Integrated Gradients)')
    
    # Coloca a legenda para fora para não tampar as linhas
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Labels TuPyE")
    
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path, dpi=300)
    if show: plt.show() 
    plt.close()