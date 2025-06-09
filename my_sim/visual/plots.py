"""
Мини-библиотека графиков: никаких сторонних хаков, только matplotlib.
Каждая функция получает готовый pandas.DataFrame, ничего не читает с диска.
"""

import matplotlib.pyplot as plt
import pandas as pd


def _with_style():
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.4)
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
def plot_velocity(df: pd.DataFrame) -> None:
    """Суммарная Velocity (SP) по спринтам."""
    vel = df.groupby("sprint")["sp_done"].sum()

    plt.figure()
    vel.plot(marker="o")
    plt.title("Velocity per Sprint")
    plt.xlabel("Sprint #")
    plt.ylabel("Story-points")
    _with_style()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
def plot_tp_hist(df: pd.DataFrame) -> None:
    """Гистограмма усталости в конце тиков."""
    plt.figure()
    df["tp_end"].hist(bins=25)
    plt.title("Distribution of Tire-Points (tp)")
    plt.xlabel("tp")
    plt.ylabel("Frequency")
    _with_style()
    plt.show()
