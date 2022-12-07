from pathlib import Path

import init_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from init_path import parent_path

FOLDER = Path(parent_path) / "results" / "report"
FOLDER.mkdir(parents=True, exist_ok=True)

ATTACKS = ["FFGSM", "PGD", "PGDL2", "TIFGSM", "TPGD"]
TRAINING_CODE = {"U": "Regular Training", "A": "Augmented Training"}

TTA_CODE = {
    "U": "No Test Time Augmentations",
    "N": "Noise Time Augmentations",
    "A": "All Time Augmentations",
}

NOISE = "Noise"
TRANS = "Translation"
ROTAT = "Rotation"


def get_e1_accuracies(df, model, num_attacks):
    data = df[df["Model"] == model]
    acc_col = data["Accuracy"].to_numpy()
    acc_matrix = acc_col.reshape((-1, num_attacks))
    acc_mean = np.mean(acc_matrix, axis=0)
    acc_std = np.std(acc_matrix, axis=0)

    return acc_mean, acc_std


def get_e2_acc(df, model, is_adv=True, augmentation="Noise", num_strengths=1):
    data = df[df["Model"] == model]
    data = data[data["Adversarial"] == is_adv]
    data = data[data["Augmentation"] == augmentation]
    acc_col = data["Metric"].to_numpy()

    acc_matrix = acc_col.reshape((-1, num_strengths))
    acc_mean = np.mean(acc_matrix, axis=0)
    acc_std = np.std(acc_matrix, axis=0)

    return acc_mean, acc_std


def get_e1_attack_intensity(df: pd.DataFrame):
    return df["Attack Intensity"].unique()


def get_e2_augmentation_strength(df: pd.DataFrame, augmentation="Noise"):
    data = df[df["Augmentation"] == augmentation]
    strengths = data["Intensity"].unique()
    return strengths


def plot_a1(e1_df, attack):
    """Do additional augmentations help?
    Fix model training, try different test time augmentations
    """
    # U for non-robust. A for robust
    for model in ["U", "A"]:
        attack_int = get_e1_attack_intensity(e1_df)
        defall_acc, defall_acc_std = get_e1_accuracies(
            e1_df, model + "A", num_attacks=len(attack_int)
        )
        defnoise_acc, defnoise_acc_std = get_e1_accuracies(
            e1_df, model + "N", num_attacks=len(attack_int)
        )
        ndef_acc, ndef_acc_std = get_e1_accuracies(
            e1_df, model + "U", num_attacks=len(attack_int)
        )

        plt.xlabel(f"{attack} Attack Strenghs")
        plt.ylabel("Accuracy")
        plt.errorbar(
            attack_int,
            defall_acc,
            yerr=defall_acc_std,
            label="w/ All Test Time Augmentations",
        )
        plt.errorbar(
            attack_int,
            defnoise_acc,
            yerr=defnoise_acc_std,
            label="w/ Noise Test Time Augmentations",
        )
        plt.errorbar(
            attack_int, ndef_acc, yerr=ndef_acc_std, label="w/ No Test Time Augmentations"
        )
        plt.legend()

        folder = FOLDER / "a1"
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / f"model={model}_attack={attack}.png")
        plt.clf()


def plot_a2(e1_df, attack):
    """Do training help?
    Fixed test time augmentations, try different training processes
    """
    # U for no test time augmentations, A for all tta, N for noise tta
    for tta in ["N", "A", "U"]:
        attack_int = get_e1_attack_intensity(e1_df)
        num_attacks = len(attack_int)
        rob_acc, rob_acc_std = get_e1_accuracies(e1_df, "A" + tta, num_attacks)
        nrob_acc, nrob_acc_std = get_e1_accuracies(e1_df, "U" + tta, num_attacks)

        plt.title(f"w/ {TTA_CODE[tta]}")
        plt.xlabel(f"{attack} Attack Strengths")
        plt.ylabel("Accuracy")
        plt.errorbar(attack_int, rob_acc, yerr=rob_acc_std, label="w/ Augmented Training")
        plt.errorbar(attack_int, nrob_acc, yerr=nrob_acc_std, label="w/ Regular Training")
        plt.legend()

        folder = FOLDER / "a2"
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / f"tta={tta}_attack={attack}.png")
        plt.clf()


def plot_a3(e2_df, attack):
    """How robust to TTA are clean and advsarial imagse?
    Fixed type of augmentations and training processes
    try clean and advsarial accuracies
    """
    for aug_type in [NOISE, TRANS, ROTAT]:
        # U for non-robust. A for robust
        for model in ["U", "A"]:
            strengths = get_e2_augmentation_strength(e2_df, augmentation=aug_type)
            num_strengths = len(strengths)
            clean_acc, clean_acc_std = get_e2_acc(
                e2_df,
                model=model,
                is_adv=False,
                augmentation=aug_type,
                num_strengths=num_strengths,
            )
            adv_acc, adv_acc_std = get_e2_acc(
                e2_df,
                model=model,
                is_adv=True,
                augmentation=aug_type,
                num_strengths=num_strengths,
            )

            plt.title(f"Type of Augmentation & Attack: {aug_type}, {attack}")
            plt.xlabel(f"Test-time augmentation strengths")
            plt.ylabel("Accuracy")

            plt.errorbar(strengths, clean_acc, yerr=clean_acc_std, label="Clean Accuracy")
            plt.errorbar(
                strengths, adv_acc, yerr=adv_acc_std, label="Advesarial Accuracy"
            )
            plt.legend()

            folder = FOLDER / "a3"
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(folder / f"aug={aug_type}_model={model}_attack={attack}.png")
            plt.clf()


if __name__ == "__main__":

    for attack in ATTACKS:
        e1_df = pd.read_csv(parent_path + f"/Experiment1_{attack}.csv")
        e2_df = pd.read_csv(parent_path + f"/Experiment2_{attack}.csv")
        # plot_a1(e1_df, attack)
        # plot_a2(e1_df, attack)
        plot_a3(e2_df, attack)
