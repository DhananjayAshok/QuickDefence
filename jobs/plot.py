import init_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from init_path import parent_path
from pathlib import Path

FOLDER = Path(parent_path) / "results" / "slide"
FOLDER.mkdir(parents=True, exist_ok=True)
E1_DF = pd.read_csv(parent_path + "/NewExperiment1.csv")
E2_DF = pd.read_csv(parent_path + "/NewExperiment2.csv")

TRAINING_CODE = {
    "U": "Regular Training",
    "A": "Augmented Training"
}

TTA_CODE =  {
    "U": "No Test Time Augmentations",
    "N": "Noise Time Augmentations",
    "A": "All Time Augmentations",
}

NOISE = "Noise"
TRANS = "Translation"
ROTAT = "Rotation"

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

    plt.xlabel("PGD Attack l2 norm")
    plt.ylabel("Accuracy")
    plt.plot(attack_int, def_acc, label="w/ Test Time Augmentations")
    plt.plot(attack_int, ndef_acc, label="w/o Test Time Augmentations")
    plt.legend()
    plt.savefig(FOLDER / f"s2_model={model}.png")

def plot_slide3(df):
    aug = "Translation"
    model = "U" # U for non-robust. A for robust
    strengths = get_e2_augmentation_strength(df, augmentation=aug)
    clean_acc = get_e2_acc(df, model=model, is_adv=False, augmentation=aug)
    adv_acc = get_e2_acc(df, model=model, is_adv=True, augmentation=aug)
    plt.xlabel(f"Test-time {aug.lower()} augmentation strengths")
    plt.ylabel("Accuracy")
    plt.plot(strengths, clean_acc, label="Clean Accuracy")
    plt.plot(strengths, adv_acc, label="Advesarial Accuracy")
    plt.legend()
    plt.savefig(FOLDER / f"s3_model={model}-aug={aug.lower()}.png")

def plot_a1():
    """Do additional augmentations help?
    Fix model training, try different test time augmentations
    """
    model = "U" # U for non-robust. A for robust

    attack_int = get_e1_attack_intensity(E1_DF)
    defall_acc = get_e1_accuracies(E1_DF, model + "A")
    defnoise_acc = get_e1_accuracies(E1_DF, model + "N")
    ndef_acc = get_e1_accuracies(E1_DF, model + "U")

    plt.xlabel("PGD Attack l2 norm")
    plt.ylabel("Accuracy")
    plt.plot(attack_int, defall_acc, label="w/ All Test Time Augmentations")
    plt.plot(attack_int, defnoise_acc, label="w/ Noise Test Time Augmentations")
    plt.plot(attack_int, ndef_acc, label="w/ No Test Time Augmentations")
    plt.legend()

    folder = FOLDER / "a1"
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / f"model={model}.png")

def plot_a2():
    """Do training help?
    Fixed test time augmentations, try different training processes
    """
    tta = "N" # U for no test time augmentations, A for all tta, N for noise tta

    attack_int = get_e1_attack_intensity(E1_DF)
    rob_acc = get_e1_accuracies(E1_DF, "A" + tta)
    nrob_acc = get_e1_accuracies(E1_DF, "U" + tta)

    plt.title(f"w/ {TTA_CODE[tta]}")
    plt.xlabel("PGD Attack l2 norm")
    plt.ylabel("Accuracy")
    plt.plot(attack_int, rob_acc, label="w/ Augmented Training")
    plt.plot(attack_int, nrob_acc, label="w/ Regular Training")
    plt.legend()

    folder = FOLDER / "a2"
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / f"tta={tta}.png")

def plot_a3():
    """How robust to TTA are clean and advsarial imagse?
    Fixed type of augmentations and training processes
    try clean and advsarial accuracies
    """
    aug_type = TRANS
    model = "A" # U for non-robust. A for robust

    for aug_type in [NOISE, TRANS, ROTAT]:
        for model in ["U", "A"]:
            clean_acc = get_e2_acc(E2_DF, model=model, is_adv=False, augmentation=aug_type)
            adv_acc = get_e2_acc(E2_DF, model=model, is_adv=True, augmentation=aug_type)
            strengths = get_e2_augmentation_strength(E2_DF, augmentation=aug_type)

            plt.title(f"Type of Augmentation: {aug_type}")
            plt.xlabel(f"Test-time augmentation strengths")
            plt.ylabel("Accuracy")

            plt.plot(strengths, clean_acc, label="Clean Accuracy")
            plt.plot(strengths, adv_acc, label="Advesarial Accuracy")
            plt.legend()

            folder = FOLDER / "a3"
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(folder / f"aug={aug_type}_model={model}.png")
            plt.clf()


if __name__ == "__main__":

    # plot_slide1(E1_DF)
    # plot_slide2(E1_DF)
    # plot_slide3(E2_DF)
    # plot_a1()
    plot_a2()