import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# citljiv izgled svih grafova
plt.rcParams.update({
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "legend.fontsize": 8,
})


ID_COLS = ["iso3", "country_name", "capital"]

BINARY_COLS = [
    "landlocked", "island_nation",
    "un_member", "eu_member", "oecd_member", "nato_member",
    "g20_member", "commonwealth_member", "opec_member",
]

CORE_CONTINUOUS = [
    "gdp_per_capita_ppp", "unemployment_rate",
    "industry_pct_gdp", "services_pct_gdp", "agriculture_pct_gdp",
    "rd_expenditure_pct_gdp", "inflation_pct",
    "population_total", "population_density", "population_growth_pct",
    "urban_population_pct", "fertility_rate", "pop_65_plus_pct",
    "life_expectancy", "infant_mortality",
    "health_expenditure_pct_gdp", "education_expenditure_pct_gdp",
    "internet_users_pct", "electricity_access_pct", "land_area_km2",
]

LOG_SCALE_VARS = [
    "gdp_per_capita_ppp", "population_total",
    "population_density", "land_area_km2"
]

CORE_CATEGORICAL = [
    "continent", "region", "hemisphere", "primary_language_family",
    "income_group", "economic_structure", "gdp_tier", "population_tier",
    "density_tier", "urbanization_tier", "fertility_tier", "aging_tier",
    "life_expectancy_tier", "internet_tier",
]

# Hrvatski nazivi skupina za naslove (vrijednosti u podacima ostaju na engleskom)
GRUPA_OPIS = {
    "income_group": "dohodovnoj skupini",
    "continent": "kontinentu",
}
GRUPA_DATOTEKA = {
    "income_group": "07_boxplot_dohodak.png",
    "continent": "08_boxplot_kontinent.png",
}


def filtriraj_stupce(stupci, df):
    return [s for s in stupci if s in df.columns]

def spremi(fig, izlazni_dir, ime_datoteke):
    fig.tight_layout()
    fig.savefig(izlazni_dir / ime_datoteke, dpi=140, bbox_inches="tight")
    plt.close(fig)

def ucitaj_podatke(putanja):
    if not putanja.exists():
        sys.exit(f"Greška: datoteka nije pronađena - {putanja}")
    return pd.read_csv(putanja)


def statistika(df, izlazni_dir):
    num_stupci = filtriraj_stupce(CORE_CONTINUOUS, df)
    opis = df[num_stupci].describe().T
    opis["asimetrija"] = df[num_stupci].skew()
    opis["kurtosis"] = df[num_stupci].kurtosis()
    opis.round(4).to_csv(izlazni_dir / "numericka_statistika.csv")

    kat_stupci = filtriraj_stupce(CORE_CATEGORICAL, df)
    redovi = []
    for s in kat_stupci:
        vc = df[s].value_counts()
        redovi.append({
            "stupac": s,
            "broj_jedinstvenih": int(df[s].nunique()),
            "najcesce": vc.index[0],
            "frekvencija": int(vc.iloc[0])
        })
    pd.DataFrame(redovi).to_csv(izlazni_dir / "kategorijska_statistika.csv", index=False)


def mreza_histograma(df, stupci, izlazni_dir, ime_datoteke, naslov, broj_stupaca=4):
    n = len(stupci)
    broj_redova = int(np.ceil(n / broj_stupaca))
    fig, axes = plt.subplots(broj_redova, broj_stupaca,
                             figsize=(4.5 * broj_stupaca, 3.1 * broj_redova))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, stupac) in enumerate(zip(axes, stupci)):
        podaci = df[stupac].dropna()
        # Jako asimetrične varijable prikazujemo na logaritamskoj osi radi čitljivosti
        if stupac in LOG_SCALE_VARS:
            podaci = podaci[podaci > 0]
            sns.histplot(podaci, kde=True, ax=ax, color="#2980b9",
                         log_scale=True, edgecolor="white", linewidth=0.4)
            ax.set_title(f"{stupac}  (log skala)", fontsize=9)
            ax.set_xlabel("vrijednost (log)", fontsize=8)
        else:
            sns.histplot(podaci, kde=True, ax=ax, color="#2980b9",
                         edgecolor="white", linewidth=0.4)
            ax.set_title(stupac, fontsize=9)
            ax.set_xlabel("vrijednost", fontsize=8)
        # y-os = broj država (frekvencija) — označavamo samo lijevi stupac mreže
        if i % broj_stupaca == 0:
            ax.set_ylabel("broj država", fontsize=8)
        else:
            ax.set_ylabel("")
        sns.despine(ax=ax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(naslov + "\n(stupci = broj država u rasponu; krivulja = procjena gustoće)",
                 fontsize=14)
    spremi(fig, izlazni_dir, ime_datoteke)


def distribucije(df, izlazni_dir):
    num_stupci = filtriraj_stupce(CORE_CONTINUOUS, df)
    mreza_histograma(df, num_stupci, izlazni_dir, "01_histogrami.png",
                     "Distribucije kontinuiranih varijabli")

    # Standardizirani box-plotovi: sve varijable na zajedničkoj z-skali
    # radi usporedbe raspršenosti i netipičnih vrijednosti (outliera).
    z = (df[num_stupci] - df[num_stupci].mean()) / df[num_stupci].std()
    poredak = df[num_stupci].skew().abs().sort_values(ascending=False).index
    zm = z.melt(var_name="varijabla", value_name="z")

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.boxplot(data=zm, x="z", y="varijabla", order=poredak, ax=ax,
                color="#2980b9", fliersize=2, linewidth=0.8)
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_title("Raspršenost i netipične vrijednosti (outlieri) svih varijabli\n"
                 "(sve standardizirano na z-vrijednosti; poredano po asimetriji)")
    ax.set_xlabel("z-vrijednost (broj standardnih devijacija od prosjeka)")
    ax.set_ylabel("varijabla")
    sns.despine(ax=ax)
    spremi(fig, izlazni_dir, "02_boxplotovi.png")


def korelacije(df, izlazni_dir):
    num_stupci = filtriraj_stupce(CORE_CONTINUOUS, df)
    pearson = df[num_stupci].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(14, 12))
    maska = np.triu(np.ones_like(pearson, dtype=bool))
    sns.heatmap(pearson, mask=maska, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, annot=True, fmt=".2f", annot_kws={"size": 6.5},
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "Pearsonov koeficijent (r)"},
                ax=ax)
    ax.set_title("Pearsonova korelacijska matrica kontinuiranih varijabli\n"
                 "(crveno = pozitivna, plavo = negativna korelacija; r od -1 do +1)", pad=14)
    spremi(fig, izlazni_dir, "03_korelacije_pearson.png")

    pearson.round(4).to_csv(izlazni_dir / "korelacije_pearson.csv")


def bivarijantne(df, izlazni_dir):
    boja = "income_group" if "income_group" in df.columns else None
    kandidati = [
        ("gdp_per_capita_ppp", "life_expectancy", True),
        ("gdp_per_capita_ppp", "internet_users_pct", True),
        ("life_expectancy", "fertility_rate", False),
        ("internet_users_pct", "infant_mortality", False),
    ]

    parovi = [(x, y, logx) for x, y, logx in kandidati
              if x in df.columns and y in df.columns]
    if not parovi:
        return

    broj_stupaca = 2
    broj_redova = int(np.ceil(len(parovi) / broj_stupaca))
    fig, axes = plt.subplots(broj_redova, broj_stupaca,
                             figsize=(7 * broj_stupaca, 5 * broj_redova))
    axes = np.atleast_1d(axes).ravel()

    rucke, oznake = [], []
    for i, (ax, (x, y, logx)) in enumerate(zip(axes, parovi)):
        sns.scatterplot(data=df, x=x, y=y, hue=boja, ax=ax, s=45, alpha=0.85,
                        edgecolor="white", linewidth=0.3,
                        legend=(i == 0 and boja is not None))
        if logx:
            ax.set_xscale("log")
            ax.set_xlabel(f"{x}  (log skala)")
        else:
            ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y}  vs.  {x}")
        sns.despine(ax=ax)
        # Jedna zajednička legenda za cijelu figuru
        if i == 0 and ax.get_legend() is not None:
            rucke, oznake = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    for ax in axes[len(parovi):]:
        ax.axis("off")

    fig.suptitle("Ključni bivarijatni odnosi (boja = dohodovna skupina)")
    if rucke:
        fig.legend(rucke, oznake, title="Dohodovna skupina", loc="lower center",
                   ncol=len(oznake), bbox_to_anchor=(0.5, -0.03), frameon=False)
    spremi(fig, izlazni_dir, "04_bivarijatni_odnosi.png")


def grafovi_kategorija(df, izlazni_dir):
    kat_stupci = filtriraj_stupce(["continent", "income_group", "economic_structure",
                                   "primary_language_family", "region"], df)
    if kat_stupci:
        broj_stupaca = 2
        broj_redova = int(np.ceil(len(kat_stupci) / broj_stupaca))
        fig, axes = plt.subplots(broj_redova, broj_stupaca,
                                 figsize=(8 * broj_stupaca, 4 * broj_redova))
        axes = np.atleast_1d(axes).ravel()

        for ax, stupac in zip(axes, kat_stupci):
            df[stupac].value_counts().sort_values().plot.barh(ax=ax, color="#8e44ad")
            ax.bar_label(ax.containers[0], fontsize=8, padding=2)
            ax.set_title(stupac)
            ax.set_xlabel("broj država")
            ax.margins(x=0.12)
            sns.despine(ax=ax)

        for ax in axes[len(kat_stupci):]:
            ax.axis("off")

        fig.suptitle("Distribucije kategorijskih varijabli")
        spremi(fig, izlazni_dir, "05_kategorije.png")

    bin_stupci = filtriraj_stupce(BINARY_COLS, df)
    if bin_stupci:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[bin_stupci].sum().sort_values().plot.barh(ax=ax, color="#e67e22")
        ax.bar_label(ax.containers[0], fontsize=9, padding=2)
        ax.set_title("Binarna obilježja – broj država (vrijednost = 1)")
        ax.set_xlabel("broj država")
        ax.margins(x=0.12)
        sns.despine(ax=ax)
        spremi(fig, izlazni_dir, "06_binarno.png")


def grupirani_boxplotovi(df, izlazni_dir):
    mete = filtriraj_stupce(["life_expectancy", "gdp_per_capita_ppp",
                             "internet_users_pct", "fertility_rate"], df)

    for grupa in ["income_group", "continent"]:
        if grupa not in df.columns or not mete:
            continue

        broj_stupaca = 2
        broj_redova = int(np.ceil(len(mete) / broj_stupaca))
        fig, axes = plt.subplots(broj_redova, broj_stupaca,
                                 figsize=(8 * broj_stupaca, 4.5 * broj_redova))
        axes = np.atleast_1d(axes).ravel()

        for ax, m in zip(axes, mete):
            poredak = df.groupby(grupa)[m].median().sort_values().index
            sns.boxplot(data=df, x=grupa, y=m, order=poredak, ax=ax,
                        color="#5dade2", fliersize=2, linewidth=0.8)
            ax.set_title(f"{m} po {GRUPA_OPIS[grupa]}")
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            sns.despine(ax=ax)

        for ax in axes[len(mete):]:
            ax.axis("off")

        fig.suptitle(f"Indikatori po {GRUPA_OPIS[grupa]}")
        spremi(fig, izlazni_dir, GRUPA_DATOTEKA[grupa])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("podaci", nargs="?", default="countries_complete.csv")
    ap.add_argument("--izlaz", default="eda_rezultati")
    args = ap.parse_args()

    izlazni_dir = Path(args.izlaz)
    izlazni_dir.mkdir(parents=True, exist_ok=True)

    df = ucitaj_podatke(Path(args.podaci))

    statistika(df, izlazni_dir)
    distribucije(df, izlazni_dir)
    korelacije(df, izlazni_dir)
    bivarijantne(df, izlazni_dir)
    grafovi_kategorija(df, izlazni_dir)
    grupirani_boxplotovi(df, izlazni_dir)


if __name__ == "__main__":
    main()