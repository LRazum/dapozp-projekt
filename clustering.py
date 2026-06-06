import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "savefig.dpi": 140, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.titlesize": 15, "figure.titleweight": "bold",
})

SLUCAJNO_STANJE = 42
KONT_ZNACAJKE = [
    "unemployment_rate", "industry_pct_gdp", "services_pct_gdp", "agriculture_pct_gdp", 
    "rd_expenditure_pct_gdp", "inflation_pct", "population_total", "population_density",
    "population_growth_pct", "urban_population_pct", "fertility_rate", "pop_65_plus_pct", 
    "life_expectancy", "infant_mortality", "health_expenditure_pct_gdp", 
    "education_expenditure_pct_gdp", "internet_users_pct", "electricity_access_pct", "land_area_km2",
]
LOG_ZNACAJKE = ["population_total", "population_density", "land_area_km2"]
PROFIL_ZNACAJKE = [
    "agriculture_pct_gdp", "services_pct_gdp", "rd_expenditure_pct_gdp", "fertility_rate", 
    "pop_65_plus_pct", "life_expectancy", "infant_mortality", "internet_users_pct",
    "electricity_access_pct", "urban_population_pct", "population_growth_pct",
]

# OBRADA PODATAKA
def ucitaj_i_pripremi(putanja):
    if not putanja.exists(): sys.exit(f"Greška: datoteka nije pronađena - {putanja}")
    df = pd.read_csv(putanja)
    kont = [c for c in KONT_ZNACAJKE if c in df.columns]
    X = df[kont].copy()
    if X.isna().any().any(): X = X.fillna(X.median(numeric_only=True))
    for c in LOG_ZNACAJKE:
        if c in X.columns: X[c] = np.log10(X[c].clip(lower=1e-3))
    return df.reset_index(drop=True), X, StandardScaler().fit_transform(X), kont

# CRTAČI GRAFOVA
def graf_izbor_k(Xs, izlazni_dir, raspon=range(2, 11)):
    inercije, silhe = [], []
    for k in raspon:
        km = KMeans(n_clusters=k, random_state=SLUCAJNO_STANJE, n_init=20).fit(Xs)
        inercije.append(km.inertia_)
        silhe.append(silhouette_score(Xs, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(raspon), inercije, "o-", color="#2980b9")
    ax1.set(title="Metoda lakta (elbow)", xlabel="broj klastera (k)", ylabel="inertia")
    sns.despine(ax=ax1)

    ax2.plot(list(raspon), silhe, "o-", color="#16a085")
    najk = list(raspon)[int(np.argmax(silhe))]
    ax2.axvline(najk, color="#e67e22", ls="--", label=f"maksimum pri k={najk}")
    ax2.set(title="Silhouette analiza", xlabel="broj klastera (k)", ylabel="prosječni silhouette")
    ax2.legend(); sns.despine(ax=ax2)

    fig.suptitle("Izbor optimalnog broja klastera (k)"); fig.tight_layout()
    fig.savefig(izlazni_dir / "01_izbor_k.png")

def graf_dendrogram(Xs, df, izlazni_dir, k):
    Z = linkage(Xs, method="ward")
    fig, ax = plt.subplots(figsize=(24, 8))
    
    dendrogram(
        Z, 
        labels=df["iso3"].values, 
        leaf_font_size=8, 
        leaf_rotation=90, 
        color_threshold=Z[-(k - 1), 2], 
        ax=ax
    )
    
    ax.axhline(Z[-(k - 1), 2], color="#888888", ls="--", label=f"rez na k={k}")
    ax.set(title="Hijerarhijsko grupiranje (Ward) — dendrogram", xlabel="država (ISO3)", ylabel="Ward udaljenost")
    ax.legend(); fig.tight_layout(); fig.savefig(izlazni_dir / "02_dendrogram.png")

def graf_pca(Xs, oznake, df, izlazni_dir):
    pca = PCA(n_components=2, random_state=SLUCAJNO_STANJE)
    proj = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11, 8))
    paleta = sns.color_palette("Set2", len(np.unique(oznake)))
    for i, c in enumerate(sorted(np.unique(oznake))):
        m = oznake == c
        ax.scatter(proj[m, 0], proj[m, 1], s=55, color=paleta[i], alpha=0.8, edgecolor="white", label=f"klaster {c} (n={int(m.sum())})")

    istaknute = ["USA", "CHN", "IND", "DEU", "QAT", "NGA", "BRA", "JPN", "LUX", "COD"]
    for _, row in df.iterrows():
        if row["iso3"] in istaknute:
            idx = df.index[df["iso3"] == row["iso3"]][0]
            ax.annotate(row["iso3"], (proj[idx, 0], proj[idx, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set(title="Klasteri u prostoru prve dvije glavne komponente (PCA)", xlabel=f"PC1 ({var[0]:.0f}% varijance)", ylabel=f"PC2 ({var[1]:.0f}% varijance)")
    ax.legend(title="grupa"); sns.despine(ax=ax); fig.tight_layout(); fig.savefig(izlazni_dir / "03_pca_klasteri.png")

def graf_profili(df, oznake, izlazni_dir):
    prof_znac = [c for c in PROFIL_ZNACAJKE if c in df.columns]
    z = (df[prof_znac] - df[prof_znac].mean()) / df[prof_znac].std()
    z["klaster"] = oznake
    medijani = z.groupby("klaster").median()

    fig, ax = plt.subplots(figsize=(12, 0.7 * len(medijani) + 3))
    sns.heatmap(medijani.T, cmap="RdBu_r", center=0, annot=True, fmt=".1f", annot_kws={"size": 8}, linewidths=0.5, ax=ax)
    ax.set(title="Profili klastera — po čemu svaki arhetip odskače", xlabel="klaster", ylabel="")
    fig.tight_layout(); fig.savefig(izlazni_dir / "04_profili_klastera.png")

def graf_silhouette(Xs, oznake, izlazni_dir):
    uzorci = silhouette_samples(Xs, oznake)
    klasteri = sorted(np.unique(oznake))
    paleta = sns.color_palette("Set2", len(klasteri))

    fig, ax = plt.subplots(figsize=(9, 7))
    y_dno = 0
    for i, c in enumerate(klasteri):
        v = np.sort(uzorci[oznake == c])
        y_vrh = y_dno + len(v)
        ax.fill_betweenx(np.arange(y_dno, y_vrh), 0, v, facecolor=paleta[i], alpha=0.8)
        ax.text(-0.04, y_dno + len(v) / 2, f"klaster {c}", va="center", fontsize=9)
        y_dno = y_vrh + 8

    ax.axvline(silhouette_score(Xs, oznake), color="#e74c3c", ls="--", label="prosjek")
    ax.set(title="Silhouette dijagram po klasteru", xlabel="silhouette koeficijent", yticks=[])
    ax.legend(); sns.despine(ax=ax, left=True); fig.tight_layout(); fig.savefig(izlazni_dir / "05_silhouette.png")

def main():
    ulazna_datoteka = Path("countries_complete.csv")
    out_folder = Path("rezultati_klasteriranje")
    out_folder.mkdir(exist_ok=True)
    broj_klastera = 3

    df, X, Xs, kont = ucitaj_i_pripremi(ulazna_datoteka)
    graf_izbor_k(Xs, out_folder)

    km = KMeans(n_clusters=broj_klastera, random_state=SLUCAJNO_STANJE, n_init=20).fit(Xs)
    oznake = km.labels_

    graf_dendrogram(Xs, df, out_folder, broj_klastera)
    graf_pca(Xs, oznake, df, out_folder)
    graf_profili(df, oznake, out_folder)
    graf_silhouette(Xs, oznake, out_folder)

    df[["iso3", "country_name", "continent", "income_group"]].assign(klaster=oznake).to_csv(out_folder / "klasteri_pripadnost.csv", index=False)
    print(f"Grafovi i CSV spremljeni u: {out_folder.resolve()}")

if __name__ == "__main__":
    main()