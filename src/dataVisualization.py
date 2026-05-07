import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plotAttributions(attributions, tokens, label_name, save_path=None, show=True):

    # 1. Filter out [PAD] and stop at [SEP] for a clean graph
    filtered_data = []
    for t, a in zip(tokens, attributions):
        filtered_data.append({"Token": t, "Attribution": float(a)})
        if t == '[SEP]':
            break
    
    # 2. Convert to DataFrame (Seaborn works best with DataFrames)
    df = pd.DataFrame(filtered_data)
    df = df[df["Token"] != "[PAD]"] # Extra safety

    # 3. Setup the plot
    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid") # Clean academic look

    # 4. Create the lineplot
    plot = sns.lineplot(
        x=range(len(df)), 
        y="Attribution", 
        data=df, 
        marker='o',       # Adds dots at each token
        color='#34495e', 
        linewidth=2
    )

    # 5. Set the token names on the X-axis
    plt.xticks(range(len(df)), df["Token"], rotation=45, ha='right')
    
    # 6. Aesthetics
    plt.axhline(0, color='red', linestyle='--', alpha=0.5) # Baseline
    plt.title(f'Attribution Flow: {label_name.capitalize()}', fontsize=14)
    plt.xlabel('Tokens', fontsize=12)
    plt.ylabel('Attribution Score', fontsize=12)
    
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