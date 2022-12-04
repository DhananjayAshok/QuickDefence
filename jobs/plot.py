import init_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from init_path import parent_path
from pathlib import Path

FOLDER = Path(parent_path) / "results" / "slide"
FOLDER.mkdir(parents=True, exist_ok=True)

def get_numpy_from_string(tstr: str):
    tstr = tstr[tstr.find("(")+1:]
    tstr = tstr[:tstr.rfind(",")]
    n = eval(tstr)
    return np.array(n)

def get_e1_accuracies(df, model):
    data = df[df["Model"] == model]
    acc_col = data["Accuracy"]
    return acc_col

def get_e2_acc(df, model, is_adv=True, augmentation="Noise"):
    data = df[df["Model"] == model]
    data = data[df["Adversarial"] == is_adv]
    data = data[df["Augmentation"] == augmentation]
    return data["Accuracy"]

def get_e1_attack_intensity(df: pd.DataFrame):
    return df["Attack Intensity"].unique()

def get_e2_augmentation_strength(df: pd.DataFrame, augmentation="Noise"):
    data = df[df["Augmentation"] == augmentation]
    strengths = data["Intensity"].unique()
    return strengths

def plot_slide1(df):
    """Compare accuracy of robustly trained vs. non-robustly trained models """
    tt_def = "U" # U for non test time defence, A for test time defence
    max_attack_idx = 7

    attack_int = get_e1_attack_intensity(df)[:max_attack_idx]
    rob_acc = get_e1_accuracies(df, "A" + tt_def)[:max_attack_idx]
    nrob_acc = get_e1_accuracies(df, "U" + tt_def)[:max_attack_idx]

    print(f"{rob_acc=}")
    print(f"{nrob_acc=}")

    plt.xlabel("Attack l2 norm")
    plt.ylabel("Accuracy")
    plt.plot(attack_int, rob_acc, label="Robust Accuracy")
    plt.plot(attack_int, nrob_acc, label="Nonrobust Accuracy")
    plt.legend()
    plt.savefig(FOLDER / f"s1_ttd={tt_def}.png")

def plot_slide2(df):
    """Compare accuracy w/ or w/o test time defence"""
    model = "A" # U for non-robust. A for robust

    attack_int = get_e1_attack_intensity(df)
    def_acc = get_e1_accuracies(df, model + "A")
    ndef_acc = get_e1_accuracies(df, model + "U")

    print(f"{def_acc=}")
    print(f"{ndef_acc=}")

    plt.xlabel("Attack l2 norm")
    plt.ylabel("Accuracy")
    plt.plot(attack_int, def_acc, label="w/ Test Time Augmentations")
    plt.plot(attack_int, ndef_acc, label="w/o Test Time Augmentations")
    plt.legend()
    plt.savefig(FOLDER / f"s2_model={model}.png")

def plot_slide3(df):
    aug = "Noise"
    model = "U" # U for non-robust. A for robust
    strengths = get_e2_augmentation_strength(df, augmentation=aug)
    clean_acc = get_e2_acc(df, model=model, is_adv=False, augmentation=aug)
    adv_acc = get_e2_acc(df, model=model, is_adv=True, augmentation=aug)
    plt.xlabel("Test-time noise augmentation strengths")
    plt.ylabel("Accuracy")
    plt.plot(strengths, clean_acc, label="Clean Accuracy")
    plt.plot(strengths, adv_acc, label="Advesarial Accuracy")
    plt.legend()
    plt.savefig(FOLDER / f"s3_model={model}.png")

if __name__ == "__main__":
    csv1 = pd.read_csv(parent_path + "/NewExperiment1.csv")
    csv2 = pd.read_csv(parent_path + "/NewExperiment2.csv")

    # plot_slide1(csv1)
    plot_slide2(csv1)
    #plot_slide3(csv2)