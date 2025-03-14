import os
import pandas as pd
import numpy as np
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, fo, dce
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
from openbabel import pybel as pb
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

Q = 0.75

rs = pd.read_csv("refined-set-v2020.csv")
rs['pki'] = -np.log10(rs.affinity)
rs = rs[rs.pdb != "5ny1"]

auc_pki_fo_cs = []
auc_le_fo_cs = []
auc_pki_fo1 = []
auc_le_fo1 = []
auc_pki_fo2 = []
auc_le_fo2 = []
auc_pki_dce_hs = []
auc_le_dce_hs = []
auc_pki_dce_cs = []
auc_le_dce_cs = []

for uniprot, group in rs.groupby("uniprot"):

    min_pdb = list(group.pdb[group.molecular_weigth == min(group.molecular_weigth)])[0]
    prefix = f"refined-set-aligned/{uniprot}_{min_pdb}"
    if not os.path.exists(f"{prefix}_atlas.pdb"):
        continue

    fig, axd = plt.subplot_mosaic([
        ["ROC_HS", "ROC_CS"],
    ])
    fig.suptitle(uniprot)

    #
    # Load FTMap
    #
    pm.reinitialize()
    ret = load_ftmap(f"{prefix}_atlas.pdb", "FTMap", k15_max_length=5)
    pm.load(f"refined-set-aligned/{uniprot}_{min_pdb}_ligand.pdb", f"LIG")

    #
    # Rank hotspots
    #
    hotspots = ret.kozakov2015
    hotspots = list(sorted(hotspots, key=lambda hs: (-hs.strength0, -hs.strength)))
    if len(hotspots) == 0:
        continue
    top_hs = hotspots[0]
    for hs in hotspots:
        if fo("LIG", hs.selection, verbose=False) != 0:
            top_hs = hs
            break
    
    #
    # Calculate AUC
    #

    fo_cs = []
    fo1 = []
    fo2 = []
    dce_hs = []
    dce_cs = []
    pki = []
    le = []
    for i, row in group.iterrows():
        pm.load(f"refined-set-aligned/{uniprot}_{row.pdb}_ligand.pdb", f"LIG")

        fo_cs_ = fo("LIG", "FTMap.CS_*", verbose=False)
        fo1_ = fo(top_hs.selection, "LIG", verbose=False)
        fo2_ = fo("LIG", top_hs.selection, verbose=False)
        dce_hs_ = dce("LIG", top_hs.selection, verbose=False)
        dce_cs_ = dce("LIG", "FTMap.CS_*", verbose=False)
        nha = pm.count_atoms("LIG")
        pm.delete("LIG")

        if fo1_ < 0.2:
            continue

        fo_cs.append(fo_cs_)
        fo1.append(fo1_)
        fo2.append(fo2_)
        dce_hs.append(dce_hs_)
        dce_cs.append(dce_cs_)
        pki.append(row.pki)
        le.append(row.pki / nha)

    fo_cs = pd.Series(fo_cs)
    fo1 = pd.Series(fo1)
    fo2 = pd.Series(fo2)
    dce_hs = pd.Series(dce_hs)
    dce_cs = pd.Series(dce_cs)
    pki = pd.Series(pki)
    le = pd.Series(le)

    if len(pki) < 5:
        continue

    def roc(ax, var, score, title, linestyle, c):
        q = var.quantile(Q)
        label = var > q
        fpr, tpr, thresholds = roc_curve(label, score)
        auc = roc_auc_score(label, score)
        ax.step(fpr, tpr, where='mid', marker='', linestyle=linestyle, label=f'{title} AUC={auc:.2}', c=c)
        return auc

    ax = axd['ROC_HS']
    ax.set_title('Prediction by Kozakov2015')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.margins(0.1)
    ax.plot([0, 1], [0, 1], c='red', linestyle='--', transform=ax.transAxes)

    
    auc_pki_fo1.append(roc(ax, pki, fo1, 'PKI~FO1', linestyle='-', c='C0'))
    auc_le_fo1.append(roc(ax, le, fo1, 'LE~FO1', linestyle='-', c='C1'))
    auc_pki_fo2.append(roc(ax, pki, fo2, 'PKI~FO2', linestyle=':', c='C0'))
    auc_le_fo2.append(roc(ax, le, fo2, 'LE~FO2', linestyle=':', c='C1'))
    auc_pki_dce_hs.append(roc(ax, pki, dce_hs, 'PKI~DCE', linestyle='-.', c='C0'))
    auc_le_dce_hs.append(roc(ax, le, dce_hs, 'LE~DCE', linestyle='-.', c='C1'))
    
    ax.legend(loc='lower right')

    ax = axd['ROC_CS']
    ax.set_title('Prediction by CS')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.margins(0.1)
    ax.plot([0, 1], [0, 1], c='red', linestyle='--', transform=ax.transAxes)

    auc_pki_fo_cs.append(roc(ax, pki, fo_cs, 'PKI~FO', linestyle='-', c='C0'))
    auc_le_fo_cs.append(roc(ax, le, fo_cs, 'LE~FO', linestyle='-', c='C1'))
    auc_pki_dce_cs.append(roc(ax, pki, dce_cs, 'PKI~DCE', linestyle='-.', c='C0'))
    auc_le_dce_cs.append(roc(ax, le, dce_cs, 'LE~DCE', linestyle='-.', c='C1'))

    ax.legend(loc='lower right')

    plt.tight_layout()
    if uniprot == "O60885":
        plt.show()
    plt.savefig(f"pics/ROC_SPECIFIC_{uniprot}.png")
    plt.close()


def freqpoly(x, *, ax, **kwargs):
    counts, bins = np.histogram(x, range=(0, 1))
    ax.plot(bins[:-1], counts, **kwargs)
    ax.yaxis.set_major_formatter(lambda x, pos: int(x))
    ax.set_ylabel("Count")


fig, (axd) = plt.subplot_mosaic([
    ['AUC_HS', 'AUC_CS']
])
fig.suptitle(
    "Affinity prediction performance\n"
    "(for S0/S ranked hotspots)"
)

ax = axd['AUC_HS']
ax.set_title("Kozakov2015 Performance")
freqpoly(auc_pki_fo1, ax=ax, label='PKI~FO1', linestyle='-', color='C0')
freqpoly(auc_le_fo1, ax=ax, label='LE~FO1', linestyle='-',  color='C1')
freqpoly(auc_pki_fo2, ax=ax, label='PKI~FO2', linestyle=':', color='C0')
freqpoly(auc_le_fo2, ax=ax, label='LE~FO2', linestyle=':',  color='C1')
freqpoly(auc_pki_dce_hs, ax=ax, label='PKI~DCE', linestyle='-.', color='C0')
freqpoly(auc_le_dce_hs, ax=ax, label='LE~DCE', linestyle='-.',  color='C1')
ax.set_xlabel("AUC")
ax.legend(loc='upper left')

ax = axd['AUC_CS']
ax.set_title("CS Performance")
freqpoly(auc_pki_fo_cs, ax=ax, label='PKI~FO', linestyle='-', color='C0')
freqpoly(auc_le_fo_cs, ax=ax, label='LE~FO', linestyle='-',  color='C1')
freqpoly(auc_pki_dce_cs, ax=ax, label='PKI~DCE', linestyle='-.', color='C0')
freqpoly(auc_le_dce_cs, ax=ax, label='LE~DCE', linestyle='-.',  color='C1')
ax.set_xlabel("AUC")
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('pics/ROC_GENERAL.png')
plt.show()
plt.close()
