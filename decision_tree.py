"""
02_stablo_odlucivanja.py
Metoda #1 — Stablo odlučivanja (Decision Tree).

Problem: klasifikacija države u dohodovnu skupinu Svjetske banke
(income_group: Low / Lower middle / Upper middle / High) NA TEMELJU
demografskih, zdravstvenih, infrastrukturnih i strukturnih pokazatelja.

Ključna metodološka odluka — izbjegavanje curenja informacija (data leakage):
income_group je po definiciji izveden iz nacionalnog dohotka, pa bi uključivanje
gdp_per_capita_ppp (ili gdp_tier) značilo da model trivijalno "prepisuje" ciljnu
varijablu. Zato te stupce ISKLJUČUJEMO, kao i sve *_tier diskretizacije i
economic_structure (koji je i sam pravilo nad udjelima industrije/usluga/poljopr.).
Time pitanje postaje sadržajno: koliko se dohodovni status može rekonstruirati iz
razvojnih pokazatelja koji nisu izravna mjera dohotka.

Izlaz: 3 grafa (stablo, matrica zabune, važnost značajki) + tekstualni izvještaj
s metrikama i izlučenim pravilima.

Pokretanje:
    python 02_stablo_odlucivanja.py
    python 02_stablo_odlucivanja.py countries_complete.csv --izlaz rezultati_stablo
"""
import argparse # parser naredbenog retka
import sys
from pathlib import Path # rad s datotekama i direktorijima

import matplotlib
matplotlib.use("Agg") # crtanje bez sučelja, samo sprema png datoteke 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, f1_score)
from sklearn.dummy import DummyClassifier # usporedba je li model bolji od trivijalnog "najčešće klase"
from sklearn.inspection import permutation_importance # pouzdanija mjera važnosti značajki od Gini kad su korelirane, permutacijom

# izgled grafova
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

SLUCAJNO_STANJE = 42 # za ponovljivost rezultata

CILJ = "income_group"
# Poredak klasa od najniže prema najvišoj (za čitljivost matrice zabune)
PORE_KLASA = ["Low income", "Lower middle income",
              "Upper middle income", "High income"]

# Kontinuirane značajke — BEZ gdp_per_capita_ppp (curenje informacija).
KONT_ZNACAJKE = [
    "unemployment_rate", "industry_pct_gdp", "services_pct_gdp",
    "agriculture_pct_gdp", "rd_expenditure_pct_gdp", "inflation_pct",
    "population_total", "population_density", "population_growth_pct",
    "urban_population_pct", "fertility_rate", "pop_65_plus_pct",
    "life_expectancy", "infant_mortality", "health_expenditure_pct_gdp",
    "education_expenditure_pct_gdp", "internet_users_pct",
    "electricity_access_pct", "land_area_km2",
]

# Binarna obilježja (geografija + članstva)
BIN_ZNACAJKE = [
    "landlocked", "island_nation", "un_member", "eu_member", "oecd_member",
    "nato_member", "g20_member", "commonwealth_member", "opec_member",
]

# Nominalne značajke (one-hot). region/primary_language_family namjerno
# izostavljeni zbog visoke kardinalnosti na malom uzorku (~200 redaka).
NOM_ZNACAJKE = ["continent", "hemisphere"]


def ucitaj_i_pripremi(putanja):
    if not putanja.exists():
        sys.exit(f"Greška: datoteka nije pronađena - {putanja}")
    df = pd.read_csv(putanja)

    # Zadržavamo samo redove sa standardnom dohodovnom skupinom, izbaci Not classified
    df = df[df[CILJ].isin(PORE_KLASA)].copy()

    # sigurnosna provjera da uzmemo samo stupce koje želimo
    kont = [c for c in KONT_ZNACAJKE if c in df.columns]
    binc = [c for c in BIN_ZNACAJKE if c in df.columns]
    nom = [c for c in NOM_ZNACAJKE if c in df.columns]

    # numerički i binarni --> matrica značajki, nominalni --> one-hot encoding
    X_num = df[kont + binc].astype(float)
    X_cat = pd.get_dummies(df[nom], prefix=nom).astype(int) # kategorijske u 0/1 (One-Hot Encoding)
    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat.reset_index(drop=True)], axis=1) # matrica značajki
    y = df[CILJ].reset_index(drop=True)

    # Sigurnosna provjera — skup je već imputiran, ali za svaki slučaj, popunjavanje Na medijanom 
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # vraća matricu značajki, ciljnu varijablu i originalni DataFrame 
    return X, y, df.reset_index(drop=True)


def treniraj_model(X_train, y_train):
    # Plitko stablo namjerno: interpretabilnost je glavni cilj ove metode.
    # Prostor hiperparametara za testiranje kako bismo pronašli najbolju kombinaciju, isprobavamo sve navedene
    mreza = {
        "criterion": ["gini", "entropy"], # mjerenje čistoće
        "max_depth": [2, 3, 4, 5], # koliko duboko smije ići
        "min_samples_leaf": [1, 3, 5, 8], # min broj uzoraka u listu
        "class_weight": [None, "balanced"], # kako trenira neujednačene klase, balanced daje težinu rijetkim klasama
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SLUCAJNO_STANJE) # 5-struka cross-validation
    pretraga = GridSearchCV(
        DecisionTreeClassifier(random_state=SLUCAJNO_STANJE),
        param_grid=mreza, cv=cv, scoring="f1_macro", n_jobs=-1,
    ) # testira sve kombinacije hiperparametara i ocjenjuje ih na temelju F1-macro metrika
    pretraga.fit(X_train, y_train)

    # vraća model s najboljim hiperparametrima 
    return pretraga


def graf_stabla(model, znacajke, izlazni_dir):
    dubina = model.get_depth()
    fig, ax = plt.subplots(figsize=(max(14, 2.4 * 2 ** min(dubina, 4)), 9))
    plot_tree(model, feature_names=list(znacajke), class_names=list(model.classes_),
              filled=True, rounded=True, proportion=True, impurity=False,
              fontsize=8, ax=ax)
    ax.set_title("Stablo odlučivanja — predikcija dohodovne skupine\n"
                 "(svaki čvor: uvjet podjele, udio uzorka, raspodjela klasa i predviđena klasa; "
                 "lijevo = uvjet točan, desno = netočan)")
    fig.tight_layout()
    fig.savefig(izlazni_dir / "01_stablo.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def graf_matrice_zabune(y_test, y_pred, izlazni_dir):
    cm = confusion_matrix(y_test, y_pred, labels=PORE_KLASA)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=PORE_KLASA)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Matrica zabune (testni skup)")
    ax.set_xlabel("Predviđena klasa")
    ax.set_ylabel("Stvarna klasa")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(izlazni_dir / "02_matrica_zabune.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def graf_vaznosti(model, X_test, y_test, znacajke, izlazni_dir):
    # Gini važnost (iz modela) vs. permutacijska važnost (na testnom skupu).
    # Permutacijska je pouzdanija kad su značajke korelirane (npr. razvojni "snop").
    gini = pd.Series(model.feature_importances_, index=znacajke)
    gini = gini.sort_values(ascending=False).head(15)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=30,
        random_state=SLUCAJNO_STANJE, scoring="f1_macro",
    )
    perm_s = pd.Series(perm.importances_mean, index=znacajke)
    perm_s = perm_s.sort_values(ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    gini.sort_values().plot.barh(ax=axes[0], color="#2980b9")
    axes[0].set_title("Gini važnost (ugrađena u model)")
    axes[0].set_xlabel("relativna važnost (zbroj = 1)")
    axes[0].set_ylabel("značajka")
    sns.despine(ax=axes[0])

    perm_s.sort_values().plot.barh(ax=axes[1], color="#16a085")
    axes[1].set_title("Permutacijska važnost (na testnom skupu)")
    axes[1].set_xlabel("pad F1-macro pri nasumičnoj permutaciji značajke")
    axes[1].set_ylabel("značajka")
    sns.despine(ax=axes[1])

    fig.suptitle("Važnost značajki — usporedba dviju mjera (samo 15 najvažnijih)")
    fig.tight_layout()
    fig.savefig(izlazni_dir / "03_vaznost_znacajki.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def zapisi_izvjestaj(pretraga, y_test, y_pred, baseline_acc, baseline_f1,
                     X, znacajke, izlazni_dir):
    model = pretraga.best_estimator_
    linije = []
    linije.append("METODA #1 — STABLO ODLUČIVANJA: REZULTATI\n")
    linije.append(f"Ciljna varijabla: {CILJ}")
    linije.append(f"Broj značajki: {X.shape[1]}  |  Broj uzoraka: {X.shape[0]}")
    linije.append(f"Klase: {', '.join(PORE_KLASA)}\n")

    linije.append("Najbolji hiperparametri (GridSearchCV, f1_macro, 5-struka CV):")
    for k, v in pretraga.best_params_.items():
        linije.append(f"  {k} = {v}")
    linije.append(f"  najbolji CV f1_macro = {pretraga.best_score_:.3f}")
    linije.append(f"  dubina stabla = {model.get_depth()}  |  "
                  f"broj listova = {model.get_n_leaves()}\n")

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    linije.append("Evaluacija na testnom skupu:")
    linije.append(f"  točnost (accuracy) = {acc:.3f}")
    linije.append(f"  F1-macro           = {f1m:.3f}")
    linije.append(f"  --- referentni model (DummyClassifier, most_frequent) ---")
    linije.append(f"  baseline točnost   = {baseline_acc:.3f}")
    linije.append(f"  baseline F1-macro  = {baseline_f1:.3f}\n")

    linije.append("Izvještaj po klasama (testni skup):")
    linije.append(classification_report(y_test, y_pred, labels=PORE_KLASA,
                                         zero_division=0))

    linije.append("\nIzlučena pravila (export_text):")
    linije.append(export_text(model, feature_names=list(znacajke)))

    tekst = "\n".join(linije)
    (izlazni_dir / "rezultati_stablo.txt").write_text(tekst, encoding="utf-8")
    return tekst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("podaci", nargs="?", default="countries_complete.csv")
    ap.add_argument("--izlaz", default="rezultati_stablo")
    args = ap.parse_args()

    izlazni_dir = Path(args.izlaz)
    izlazni_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = ucitaj_i_pripremi(Path(args.podaci))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SLUCAJNO_STANJE)

    pretraga = treniraj_model(X_train, y_train)
    model = pretraga.best_estimator_
    y_pred = model.predict(X_test)

    # Referentni (baseline) model
    dummy = DummyClassifier(strategy="most_frequent", random_state=SLUCAJNO_STANJE)
    dummy.fit(X_train, y_train)
    d_pred = dummy.predict(X_test)
    baseline_acc = accuracy_score(y_test, d_pred)
    baseline_f1 = f1_score(y_test, d_pred, average="macro")

    graf_stabla(model, X.columns, izlazni_dir)
    graf_matrice_zabune(y_test, y_pred, izlazni_dir)
    graf_vaznosti(model, X_test, y_test, X.columns, izlazni_dir)

    tekst = zapisi_izvjestaj(pretraga, y_test, y_pred, baseline_acc,
                             baseline_f1, X, X.columns, izlazni_dir)
    print(tekst)
    print(f"\n[Grafovi i izvještaj spremljeni u: {izlazni_dir.resolve()}]")


if __name__ == "__main__":
    main()