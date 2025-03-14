import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sb
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, fo, dce
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import permutations, chain, product
from pprint import pprint
from scipy.stats import kendalltau
from matplotlib import pyplot as plt

Q = 0.75

rs = pd.read_csv("refined-set-v2020.csv")
rs['pki'] = -np.log10(rs.affinity)


def eval_rank(target, rank, max_length):

    auc_fo1_list = []
    auc_fo2_list = []
    auc_dce_list = []

    pkl_fname = f"calc_rank.pkl"
    if os.path.exists(pkl_fname):
        with open(pkl_fname, "rb") as f:
            rank_data, not_todo = pickle.load(f)
    else:
        rank_data, not_todo = {}, set()

    group_idx = 0
    for uniprot, group in rs.groupby("uniprot"):
        key = (target, rank, uniprot, max_length)
        try:
            auc_fo1, auc_fo2, auc_dce = rank_data.get(key, (None, None))
        except Exception:
            auc_fo1 = None
            auc_fo2 = None
            auc_dce = None
        
        if auc_fo1 is not None:
            auc_fo1_list.append(auc_fo1)
            auc_fo2_list.append(auc_fo2)
            auc_dce_list.append(auc_dce)
            continue

        if key in not_todo:
            continue
        
        ref_pdb = list(
            group.pdb[group.molecular_weigth == min(group.molecular_weigth)]
        )[0]
        prefix = f"refined-set-aligned/{uniprot}_{ref_pdb}"
        if not os.path.exists(f"{prefix}_atlas.pdb"):
            not_todo.add(key)
            continue
        
        # group_idx += 1
        # if group_idx >= 15:
        #     break

        # if uniprot not in ["A7YT55", "A4TVL0", "P49773", "P00918"]:
        #     continue

        pm.reinitialize()
        ret = load_ftmap(
            f"{prefix}_atlas.pdb",
            ref_pdb,
        )

        for i, row in group.iterrows():
            pm.load(
                f"refined-set-aligned/{uniprot}_{row.pdb}_ligand.pdb", f"LIG_{row.pdb}"
            )

        def sort_func(hs):
            d = {
                "S": -hs.strength,
                "S0": -hs.clusters[0].strength,
                "CD": hs.center_center,
                "MD": -hs.max_dist,
            }
            return [d[r] for r in rank]

        hotspots = ret.kozakov2015
        hotspots = list(sorted(hotspots, key=sort_func))
        if len(hotspots) == 0:
            not_todo.add(key)
            continue
        top_hs_ = hotspots[0]
        for hs in hotspots:
            if fo(f"%LIG_{ref_pdb}", hs.selection, verbose=False) != 0:
                top_hs_ = hs
                break
        fo1_lig_hs = []
        fo2_lig_hs = []
        dce_lig_hs = []
        pki = []
        le = []
        for i, row in group.iterrows():
            lig_sel = f"%LIG_{row.pdb}"
            hs_sel = top_hs_.selection
            fo1_lig_hs_ = fo(hs_sel, lig_sel, verbose=False)
            fo2_lig_hs_ = fo(lig_sel, hs_sel, verbose=False)
            dce_lig_hs_ = dce(lig_sel, hs_sel, verbose=False)

            nha = pm.count_atoms(lig_sel)

            if fo1_lig_hs_ == 0:
                continue

            fo1_lig_hs.append(fo1_lig_hs_)
            fo2_lig_hs.append(fo2_lig_hs_)
            dce_lig_hs.append(dce_lig_hs_)
            pki.append(row.pki)
            le.append(row.pki / nha)

        if len(pki) < 5:
            not_todo.add(key)
            continue

        fo1_lig_hs = pd.Series(fo1_lig_hs)
        fo2_lig_hs = pd.Series(fo2_lig_hs)
        dce_lig_hs = pd.Series(dce_lig_hs)
        pki = pd.Series(pki)
        le = pd.Series(le)
        
        if target == "PKI":
            label = pki > pki.quantile(Q)
        elif target == "LE":
            label = le > le.quantile(Q)
        else:
            raise Exception

        score = fo1_lig_hs
        fpr, tpr, thresholds = roc_curve(label, score)
        auc_fo1 = roc_auc_score(label, score)
        auc_fo1_list.append(auc_fo1)

        score = fo2_lig_hs
        fpr, tpr, thresholds = roc_curve(label, score)
        auc_fo2 = roc_auc_score(label, score)
        auc_fo2_list.append(auc_fo2)

        score = dce_lig_hs
        fpr, tpr, thresholds = roc_curve(label, score)
        auc_dce = roc_auc_score(label, score)
        auc_dce_list.append(auc_dce)

        rank_data[key] = (auc_fo1, auc_fo2, auc_dce)

    with open(pkl_fname, "wb") as f:
        pickle.dump((rank_data, not_todo), f)

    return (
        np.average(auc_fo1_list),
        np.average(auc_fo2_list),
        np.average(auc_dce_list)
    )


lengths = [3, 4, 5]
vars = ["S", "S0", "CD", "MD"]
targets = ["PKI", "LE"]

R = {}
for target in targets:
    for rank in permutations(vars, 2):
        for max_length in lengths:
            key = (target, rank, max_length)
            auc_fo1, auc_fo2, auc_dce = eval_rank(*key)
            R[key] = auc_fo1, auc_fo2, auc_dce
            print(f"{key}: {auc_fo1:.2f} {auc_fo2:.2f} {auc_dce:.2f}")

# R={('PKI', ('S', 'S0'), 3): (0.4584956763340821, 0.4844703982546952, 0.4313904185180047), ('PKI', ('S', 'S0'), 4): (0.4805427890970224, 0.49221388031422575, 0.4480017859196193), ('PKI', ('S', 'S0'), 5): (0.4725758935502991, 0.49221701337261703, 0.43716630645200705), ('PKI', ('S', 'S0'), 6): (0.4724487641325518, 0.49247127220811154, 0.43767482412299613), ('PKI', ('S', 'CD'), 3): (0.4592304932978566, 0.49234045494057527, 0.43201549009108553), ('PKI', ('S', 'CD'), 4): (0.4805427890970224, 0.49221388031422575, 0.4480017859196193), ('PKI', ('S', 'CD'), 5): (0.4725758935502991, 0.49221701337261703, 0.43716630645200705), ('PKI', ('S', 'CD'), 6): (0.4724487641325518, 0.49247127220811154, 0.43767482412299613), ('PKI', ('S', 'MD'), 3): (0.45538691041015605, 0.48206304852346293, 0.42967591789857223), ('PKI', ('S', 'MD'), 4): (0.48137890949836015, 0.49277129391511754, 0.44744437231872747), ('PKI', ('S', 'MD'), 5): (0.4725758935502991, 0.49221701337261703, 0.43716630645200705), ('PKI', ('S', 'MD'), 6): (0.4724487641325518, 0.49247127220811154, 0.43767482412299613), ('PKI', ('S0', 'S'), 3): (0.4587998963827573, 0.4845277982638792, 0.4299094982810574), ('PKI', ('S0', 'S'), 4): (0.47152582997314957, 0.4902000550644775, 0.4386058203582938), ('PKI', ('S0', 'S'), 5): (0.4708688759468942, 0.49207834703464093, 0.43824439236397017), ('PKI', ('S0', 'S'), 6): (0.4707417465291469, 0.49233260587013544, 0.43875291003495925), ('PKI', ('S0', 'CD'), 3): (0.46269638329220375, 0.47542038509680873, 0.4477824006134883), ('PKI', ('S0', 'CD'), 4): (0.46269638329220375, 0.47542038509680873, 0.4477824006134883), ('PKI', ('S0', 'CD'), 5): (0.46269638329220375, 0.47542038509680873, 0.4477824006134883), ('PKI', ('S0', 'CD'), 6): (0.46269638329220375, 0.47542038509680873, 0.4477824006134883), ('PKI', ('S0', 'MD'), 3): (0.46264219565457315, 0.4839003583026566, 0.43326821552842476), ('PKI', ('S0', 'MD'), 4): (0.4707080527398346, 0.49439886139113176, 0.44601979350957577), ('PKI', ('S0', 'MD'), 5): (0.47660942600859957, 0.492967334273769, 0.44708573479025626), ('PKI', ('S0', 'MD'), 6): (0.4764822965908523, 0.49322159310926356, 0.4475942524612453), ('PKI', ('CD', 'S'), 3): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S'), 4): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S'), 5): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S'), 6): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S0'), 3): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S0'), 4): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S0'), 5): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'S0'), 6): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'MD'), 3): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'MD'), 4): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'MD'), 5): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('CD', 'MD'), 6): (0.46487823406122236, 0.49746625981350345, 0.4551305691979715), ('PKI', ('MD', 'S'), 3): (0.46314860684494985, 0.4917331779336467, 0.4341183544085735), ('PKI', ('MD', 'S'), 4): (0.46630569694267643, 0.49709842189469305, 0.4378700924596359), ('PKI', ('MD', 'S'), 5): (0.47424114089539743, 0.4961754124483194, 0.4390631631580637), ('PKI', ('MD', 'S'), 6): (0.4741140114776502, 0.49642967128381393, 0.4395716808290528), ('PKI', ('MD', 'S0'), 3): (0.46314860684494985, 0.4917331779336467, 0.4341183544085735), ('PKI', ('MD', 'S0'), 4): (0.46630569694267643, 0.49709842189469305, 0.4378700924596359), ('PKI', ('MD', 'S0'), 5): (0.47424114089539743, 0.4961754124483194, 0.4390631631580637), ('PKI', ('MD', 'S0'), 6): (0.4741140114776502, 0.49642967128381393, 0.4395716808290528), ('PKI', ('MD', 'CD'), 3): (0.46314860684494985, 0.4917331779336467, 0.4341183544085735), ('PKI', ('MD', 'CD'), 4): (0.46630569694267643, 0.49709842189469305, 0.4378700924596359), ('PKI', ('MD', 'CD'), 5): (0.47424114089539743, 0.4961754124483194, 0.4390631631580637), ('PKI', ('MD', 'CD'), 6): (0.4741140114776502, 0.49642967128381393, 0.4395716808290528), ('LE', ('S', 'S0'), 3): (0.7057610320606648, 0.39895577106546465, 0.7119847401109116), ('LE', ('S', 'S0'), 4): (0.6959800541631914, 0.3928193353257269, 0.6970166388426754), ('LE', ('S', 'S0'), 5): (0.7001919124009826, 0.39381396708456184, 0.7078025238082303), ('LE', ('S', 'S0'), 6): (0.7001919124009826, 0.3938775317934355, 0.7075482649727357), ('LE', ('S', 'CD'), 3): (0.7023040522538166, 0.394964378921326, 0.7157470983962114), ('LE', ('S', 'CD'), 4): (0.6959800541631914, 0.3928193353257269, 0.6970166388426754), ('LE', ('S', 'CD'), 5): (0.7001919124009826, 0.39381396708456184, 0.7078025238082303), ('LE', ('S', 'CD'), 6): (0.7001919124009826, 0.3938775317934355, 0.7075482649727357), ('LE', ('S', 'MD'), 3): (0.7039751752484691, 0.3966355019159784, 0.7100652802143933), ('LE', ('S', 'MD'), 4): (0.697512941565644, 0.3940735159277336, 0.6964592252417836), ('LE', ('S', 'MD'), 5): (0.7001919124009826, 0.39381396708456184, 0.7078025238082303), ('LE', ('S', 'MD'), 6): (0.7001919124009826, 0.3938775317934355, 0.7075482649727357), ('LE', ('S0', 'S'), 3): (0.7103587727963033, 0.39972493118853025, 0.7164332408226717), ('LE', ('S0', 'S'), 4): (0.7074124722242624, 0.38943656145114974, 0.7055414552791098), ('LE', ('S0', 'S'), 5): (0.7105861068837842, 0.3899685833842933, 0.7086218849800943), ('LE', ('S0', 'S'), 6): (0.7105861068837842, 0.39003214809316694, 0.7083676261445997), ('LE', ('S0', 'CD'), 3): (0.6865412047274894, 0.4073287697940771, 0.6951146939565511), ('LE', ('S0', 'CD'), 4): (0.6865412047274894, 0.4073287697940771, 0.6951146939565511), ('LE', ('S0', 'CD'), 5): (0.6865412047274894, 0.4073287697940771, 0.6951146939565511), ('LE', ('S0', 'CD'), 6): (0.6865412047274894, 0.4073287697940771, 0.6951146939565511), ('LE', ('S0', 'MD'), 3): (0.6957773924514918, 0.3924930678838674, 0.7036006681453748), ('LE', ('S0', 'MD'), 4): (0.7046125706782945, 0.3910072952519521, 0.7008316428845365), ('LE', ('S0', 'MD'), 5): (0.708288818769769, 0.3881709316898018, 0.7066290023259797), ('LE', ('S0', 'MD'), 6): (0.708288818769769, 0.38823449639867547, 0.7063747434904851), ('LE', ('CD', 'S'), 3): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S'), 4): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S'), 5): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S'), 6): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S0'), 3): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S0'), 4): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S0'), 5): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'S0'), 6): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'MD'), 3): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'MD'), 4): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'MD'), 5): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('CD', 'MD'), 6): (0.7023146729280018, 0.4277789226843921, 0.7251898834041595), ('LE', ('MD', 'S'), 3): (0.6696233285861605, 0.39298897316701825, 0.6847866273120886), ('LE', ('MD', 'S'), 4): (0.6801553860455463, 0.3988980980005001, 0.6893413768484038), ('LE', ('MD', 'S'), 5): (0.6845308459346306, 0.39536252264073996, 0.6950116068720997), ('LE', ('MD', 'S'), 6): (0.6845308459346306, 0.39542608734961354, 0.6947573480366053), ('LE', ('MD', 'S0'), 3): (0.6696233285861605, 0.39298897316701825, 0.6847866273120886), ('LE', ('MD', 'S0'), 4): (0.6801553860455463, 0.3988980980005001, 0.6893413768484038), ('LE', ('MD', 'S0'), 5): (0.6845308459346306, 0.39536252264073996, 0.6950116068720997), ('LE', ('MD', 'S0'), 6): (0.6845308459346306, 0.39542608734961354, 0.6947573480366053), ('LE', ('MD', 'CD'), 3): (0.6696233285861605, 0.39298897316701825, 0.6847866273120886), ('LE', ('MD', 'CD'), 4): (0.6801553860455463, 0.3988980980005001, 0.6893413768484038), ('LE', ('MD', 'CD'), 5): (0.6845308459346306, 0.39536252264073996, 0.6950116068720997), ('LE', ('MD', 'CD'), 6): (0.6845308459346306, 0.39542608734961354, 0.6947573480366053)}

for target in targets:
    for max_length in lengths:
        fig, axd = plt.subplot_mosaic(
            [
                ["FO1", "FO2", "DCE"]
            ]
        )
        fig.suptitle(f"Average AUC\n(Target={target}; MaxLength={max_length})")
        H_fo1 = np.zeros(shape=(len(vars), len(vars)))
        H_fo2 = np.zeros(shape=(len(vars), len(vars)))
        H_dce = np.zeros(shape=(len(vars), len(vars)))
        for i1, var1 in enumerate(vars):
            for i2, var2 in enumerate(vars):
                for (t, (r1, r2), ml), (auc_fo1, auc_fo2, auc_dce) in R.items():
                    if t == target and r1 == var1 and r2 == var2 and ml == max_length:
                        break
                else:
                    auc_fo1, auc_fo2, auc_dce = np.nan, np.nan, np.nan
                H_fo1[i1, i2] = auc_fo1
                H_fo2[i1, i2] = auc_fo2
                H_dce[i1, i2] = auc_dce

        def plot(title, H):
            ax = axd[title]
            
            vmax = np.nanmax(H)
            vmin = np.nanmin(H)

            ax.imshow(H, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("2nd var")
            ax.set_ylabel("1st var")
            ax.set_xticks(np.arange(len(vars)), labels=vars)
            ax.set_yticks(np.arange(len(vars)), labels=vars)
            
            for i in range(len(vars)):
                for j in range(len(vars)):
                    auc = H[i, j]
                    if auc > vmin + (vmax - vmin) / 2:
                        color = "black"
                    else:
                        color = "white"
                    ax.text(j, i, f"{auc:.2f}", ha="center", va="center", color=color)

        plot("FO1", H_fo1)
        plot("FO2", H_fo2)
        plot("DCE", H_dce)
        
        fig.tight_layout()
        plt.savefig(f"pics/RANK_{target}_{max_length}.png")
        plt.show()
plt.clf()