import pandas as pd
from src.dataVisualization import plotAttributions
from src.loadDataset import getPlotIndices
from tqdm import tqdm
from src import config

full_df = pd.read_csv("ig-results/Albertina.csv")

plot_indices = getPlotIndices()

for idx in tqdm(plot_indices):

    sentence_df = full_df[full_df['text_id']==idx]
    sentence_df = sentence_df.drop(columns="Unnamed: 0")

    plotAttributions(sentence_df, save_path=f"ig-results/images/Albertina/{idx}-ig.png", show=False)

    raise