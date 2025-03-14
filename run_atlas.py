import pandas as pd
import subprocess
import os
import numpy as np


rs = pd.read_csv("refined-set-v2020.csv")
rs['pki'] = np.log10(rs.affinity)

for uniprot, group in rs.groupby("uniprot"):
    if uniprot == "000000":
        continue
    if len(group) < 10 or rs.pki.max() - rs.pki.min() < 2:
        continue
    min_pdb = list(group.pdb[group.molecular_weigth == min(group.molecular_weigth)])[0]
    prefix = f"refined-set-aligned/{uniprot}_{min_pdb}"
    if os.path.exists(f"{prefix}_atlas.pdb"):
        continue

    print()
    print("Computing", prefix)
    subprocess.check_call(f"run_atlas --prefix {prefix}_atlas {prefix}_protein.pdb", shell=True)
