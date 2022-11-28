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

def get_e2():
    return pd.read_csv("Experiment2.csv")


def plot():
    augs = list(df["Augmentation"].unique())
    for aug in augs:
        ds = df[df["Augmentation"] == aug]
        ds = ds.set_index("Severity")
        ds = ds * 100
        ds["Base Rate"] = 1/101
        sns.lineplot(data=ds)
        plt.title(f"Model Accuracies over Severity for Augmentation: {aug}")
        plt.ylabel("Percent")
        plt.show()

if __name__ == "__main__":
    df = get_e2()
    plot()



