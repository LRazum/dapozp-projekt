import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# --- POSTAVKE ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "savefig.dpi": 140, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.titlesize": 15, "figure.titleweight": "bold",
})

min_sup, max_l, min_conf, m_lift = 0.05, 4, 0.65, 1.20
cilj_var = "income_group_"

kat_podaci = ["continent", "hemisphere", "income_group", "economic_structure", "population_tier", 
              "density_tier", "urbanization_tier", "fertility_tier", "aging_tier", "life_expectancy_tier", "internet_tier"]
bin_podaci = ["landlocked", "island_nation", "un_member", "eu_member", "oecd_member", 
              "nato_member", "g20_member", "commonwealth_member", "opec_member"]

# --- PRIJEVODI ZA GRAFOVE ---
r_dict1 = {"income_group_Low income": "Low income", "income_group_Lower middle income": "Lower middle income",
           "income_group_Upper middle income": "Upper middle income", "income_group_High income": "High income"}
r_dict2 = {"Africa": "Afrika", "Asia": "Azija", "Europe": "Europa", "Americas": "Amerike", "Oceania": "Oceanija"}
r_dict3 = {
    "internet_tier_": ("internet", {"Low": "nizak", "Medium": "srednji", "High": "visok", "Very-high": "vrlo visok"}),
    "life_expectancy_tier_": ("život. vijek", {"Low": "nizak", "Medium": "srednji", "High": "visok", "Very-high": "vrlo visok"}),
    "fertility_tier_": ("fertilitet", {"Below-replacement": "ispod obnove", "Replacement": "obnova", "High": "visok", "Very-high": "vrlo visok"}),
    "aging_tier_": ("dob stan.", {"Young": "mlado", "Maturing": "sazrijeva", "Mature": "zrelo", "Aged": "staro", "Super-aged": "vrlo staro"}),
    "density_tier_": ("gustoća", {"Sparse": "rijetka", "Moderate": "umjerena", "Dense": "gusta", "Very-dense": "vrlo gusta"}),
    "urbanization_tier_": ("urbaniz.", {"Rural": "ruralna", "Mixed": "miješana", "Urban": "urbana", "Hyper-urban": "hiperurbana"}),
    "population_tier_": ("stanovn.", {"Tiny": "sitna", "Small": "mala", "Medium": "srednja", "Large": "velika", "Mega": "mega"}),
    "economic_structure_": ("gosp.", {"Agriculture-significant": "poljoprivreda", "Industry-heavy": "industrija", "Service-dominant": "usluge", "Mixed": "miješano"}),
}
r_dict4 = {"un_member": "UN član", "eu_member": "EU član", "oecd_member": "OECD član", "nato_member": "NATO član",
           "g20_member": "G20 član", "commonwealth_member": "Commonwealth", "opec_member": "OPEC član",
           "landlocked": "bez izlaza na more", "island_nation": "otočna država"}

boje = {"income_group_Low income": "#c0392b", "income_group_Lower middle income": "#e67e22",
        "income_group_Upper middle income": "#2980b9", "income_group_High income": "#16a085"}
svi_razredi = list(boje.keys())

# --- OBRADA PODATAKA ---
def ucitaj_csv(p):
    if not p.exists(): sys.exit(f"Greška: datoteka {p} ne postoji!")
    t = pd.read_csv(p)
    d1 = [pd.get_dummies(t[c].astype(str), prefix=c) for c in kat_podaci if c in t.columns]
    d1 += [(t[b] == 1).rename(b).to_frame() for b in bin_podaci if b in t.columns]
    return pd.concat(d1, axis=1).astype(bool)

def dobij_skupove(t):
    s = apriori(t, min_support=min_sup, use_colnames=True, max_len=max_l)
    s["duljina"] = s["itemsets"].apply(len)
    return s.sort_values("support", ascending=False).reset_index(drop=True)

def stvori_pravila(s):
    pr = association_rules(s, metric="confidence", min_threshold=min_conf)
    pr = pr[(pr["lift"] >= m_lift) & (pr["consequents"].apply(lambda k: len(k) == 1 and next(iter(k)).startswith(cilj_var)))].copy()
    pr["uvjeti"] = pr["antecedents"].apply(len)
    return pr.sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)

def ocisti_visak(pr):
    ok = []
    for i, red in pr.iterrows():
        visak = any(pr.loc[j, "consequents"] == red["consequents"] and 
                    pr.loc[j, "antecedents"] < red["antecedents"] and 
                    pr.loc[j, "confidence"] >= red["confidence"] - 1e-9 for j in ok)
        if not visak: ok.append(i)
    return pr.loc[ok].reset_index(drop=True)

# --- POMOĆNE FUNKCIJE ZA GRAFOVE ---
def sredi_tekst(st):
    if st.startswith("continent_"): return r_dict2.get(st[10:], st[10:])
    if st.startswith("hemisphere_"): return "sjeverna polutka" if st.endswith("Northern") else "južna polutka"
    for pref, (o, m) in r_dict3.items():
        if st.startswith(pref): return f"{o}: {m.get(st[len(pref):], st[len(pref):])}"
    return r_dict4.get(st, r_dict1.get(st, st))

def vrati_grupu(cr): return r_dict1.get(cr, cr.replace("income_group_", ""))
def u1(sk): return "  +  ".join(sredi_tekst(x) for x in sorted(sk))
def u2(sk): return "\n".join(sredi_tekst(x) for x in sorted(sk))
def daj_naslov(f, a, gl, pod): f.suptitle(gl, fontsize=14, fontweight="bold"); a.set_title(pod, fontsize=9.5, color="#555555", pad=10)

def top_k_pravila(pr, k):
    p2 = pr.copy()
    p2["cons"] = p2["consequents"].apply(lambda s: next(iter(s)))
    return pd.concat([p2[p2["cons"] == r].sort_values("lift", ascending=False).head(k) 
                      for r in svi_razredi if (p2["cons"] == r).any()]).reset_index(drop=True)

# --- CRTAČI GRAFOVA ---
def crtaj_graf1(f_skupovi, dir_out):
    br = f_skupovi["duljina"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar_label(ax.bar(br.index, br.values, color="#2980b9", edgecolor="white", width=0.7), padding=4)
    daj_naslov(fig, ax, "Broj čestih skupova stavki raste s veličinom skupa", f"Apriori uz min. oslonac = {min_sup}")
    ax.set(xlabel="broj stavki (pokazatelja) u skupu", ylabel="broj čestih skupova", xticks=br.index)
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(dir_out / "01_velicine_skupova.png")

def crtaj_graf2(prv, dir_out):
    p2 = prv.copy(); p2["cons"] = p2["consequents"].apply(lambda s: next(iter(s)))
    l1, l2 = float(p2["lift"].min()), float(p2["lift"].max())
    vel = lambda x: 70.0 if l2 <= l1 else 35.0 + 165.0 * (x - l1) / (l2 - l1)

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    for r in svi_razredi:
        pod = p2[p2["cons"] == r]
        if not pod.empty: ax.scatter(pod["support"], pod["confidence"], s=[vel(x) for x in pod["lift"]], color=boje[r], alpha=0.7, label=vrati_grupu(r))
    
    ax.axhline(min_conf, color="#999999", ls="--")
    daj_naslov(fig, ax, "Kvaliteta i raspodjela asocijacijskih pravila", "Boja = razred • Veličina = lift")
    ax.set(xlabel="oslonac (support)", ylabel="pouzdanost (confidence)")
    ax.legend(title="predviđeni razred", loc="lower right"); sns.despine(ax=ax); fig.tight_layout(); fig.savefig(dir_out / "02_support_confidence.png")

def crtaj_graf3(prv, dir_out, k_raz=4):
    od_prv = top_k_pravila(prv, k_raz)
    fig, ax = plt.subplots(figsize=(13.5, 0.55 * len(od_prv) + 2.3))
    ax.barh(np.arange(len(od_prv)), od_prv["lift"], color=[boje[c] for c in od_prv["cons"]], edgecolor="white", height=0.72)
    ax.set_yticks(np.arange(len(od_prv)), [u1(s) for s in od_prv["antecedents"]], fontsize=8.5)
    ax.invert_yaxis()
    daj_naslov(fig, ax, "Najjača pravila po dohodovnom razredu", "Prikazani su uvjeti, boja stupca = razred")
    ax.set(xlabel="lift", xlim=(0, float(od_prv["lift"].max()) * 1.30))
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(dir_out / "03_top_pravila.png")

def crtaj_graf4(prv, dir_out, k_raz=3):
    top = top_k_pravila(prv, k_raz)
    akt = [r for r in svi_razredi if (top["cons"] == r).any()]
    fig, ax = plt.subplots(figsize=(15, 4.0 * len(akt)))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    
    y_pos = {r: 0.93 - ((0.93 - 0.05) / len(akt)) * (i + 0.5) for i, r in enumerate(akt)}
    for r in akt:
        g = top[top["cons"] == r].reset_index(drop=True)
        for j, red in g.iterrows():
            y_tren = y_pos[r] + (0.070 * (j - (len(g) - 1) / 2) if len(g) > 1 else 0.0)
            ax.annotate("", xy=(0.70 - 0.035, y_pos[r]), xytext=(0.045 + 0.015, y_tren), 
                        arrowprops=dict(arrowstyle="-|>", color=boje[r], lw=2, alpha=0.6, connectionstyle="arc3,rad=0.04"))
            ax.text(0.045, y_tren, u2(red["antecedents"]) + f"\nlift {red['lift']:.1f}", fontsize=8, va="center", 
                    bbox=dict(boxstyle="round", fc="white", ec=boje[r], alpha=0.9))
        ax.scatter([0.70], [y_pos[r]], s=1700, color=boje[r], zorder=4)
        ax.text(0.745, y_pos[r], vrati_grupu(r), color=boje[r], fontsize=12, fontweight="bold", va="center")

    daj_naslov(fig, ax, "Mreža pravila za predviđanje dohodovne skupine", "Čvorište = razred")
    fig.tight_layout(); fig.savefig(dir_out / "04_mreza_pravila.png")

# --- IZVRŠAVANJE ---
def main():
    ulazna_datoteka = Path("countries_complete.csv")
    out_folder = Path("rezultati_pravila")
    out_folder.mkdir(exist_ok=True)

    t_df = ucitaj_csv(ulazna_datoteka)
    cesti = dobij_skupove(t_df)
    kon_pr = ocisti_visak(stvori_pravila(cesti))

    crtaj_graf1(cesti, out_folder)
    crtaj_graf2(kon_pr, out_folder)
    crtaj_graf3(kon_pr, out_folder)
    crtaj_graf4(kon_pr, out_folder)

    print(f"Svi grafovi su uspješno generirani u mapi: {out_folder.resolve()}")

if __name__ == "__main__":
    main()
