import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def print_results_e1(df):
    df = get_e1()
    g = df.groupby(["Dataset", "Epsilon", "Augmentation", "Metric", "SampleAgg", "BatchAgg"]).mean()
    ms = ["Mean", "Max", "Min"]
    metrics = ["Correct", "Robust", "De-adversarial"]
    for metric in metrics:
        for m1 in ms:
            for m2 in ms:
                print(f"Metric: {metric}, SampleAgg {m1}, BatchAgg {m2}")
                print(g.loc[:, :, :, metric, m1, m2])




def get_e1():
    return pd.read_csv("Experiment1.csv")




df = get_e1()



