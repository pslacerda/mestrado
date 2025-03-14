import os
from os.path import splitext

import numpy as np
import pandas as pd
from openbabel import pybel
from pymol import cmd as pm


def parse_refined_set(dir, year):
    names = pd.read_fwf(
        "%s/index/INDEX_refined_name.%s" % (dir, year),
        skiprows=6,
        header=None,
        index_col=False,
        names=["pdb", "year", "uniprot", "description"],
        dtype={
            "pdb": str,
            "year": int,
            "uniprot": str,
            "description": str,
        },
    )

    names.loc[names.uniprot == '------', 'uniprot'] = '000000'

    data = pd.read_fwf(
        "%s/index/INDEX_refined_set.%s" % (dir, year),
        skiprows=6,
        header=None,
        index_col=False,
        names=["pdb", "resolution", "year", "affinity", "_1", "_2", "ligand", "_3"],
        dtype={
            "pdb": str,
            "resolution": float,
            "year": int,
            "affinity": str,
            "ligand": str,
        },
        infer_nrows=5000,
    )
    data['ligand'] = data.ligand.replace({r"\((.*?)\)\ ?.*": r"\1"}, regex=True)
    data = data[data.affinity.str.contains("=")]
    data = data.drop(["_1", "_2", "_3"], axis=1)

    affs = data.affinity.replace(
        {r"(.*)=(([0-9]*[.])?[0-9]+)": r"\1 \2 \3"}, regex=True
    ).str.split(" ", n=2, expand=True)
    affs.columns = ["constant", "affinity", "unit"]
    affs.unit = affs.unit.str[-2:]
    affs2 = []
    for idx, row in affs.iterrows():
        if row.unit == "mM":
            affs2.append(float(row.affinity) / 1e3)
        elif row.unit == "uM":
            affs2.append(float(row.affinity) / 1e6)
        elif row.unit == "nM":
            affs2.append(float(row.affinity) / 1e9)
        elif row.unit == "pM":
            affs2.append(float(row.affinity) / 1e12)
        elif row.unit == "fM":
            affs2.append(float("nan"))
        else:
            raise NotImplementedError()

    data['affinity'] = pd.Series(affs2, name="affinity")
    data['pki'] = -np.log10(data.affinity)
    return pd.merge(names, data, on=["pdb", "year"]).dropna(axis=0)


def extract_mw(mol_path):
    _, ext = splitext(mol_path)
    mol = pybel.readfile(ext[1:], mol_path)
    mol = next(mol)
    return mol.OBMol.GetExactMass()


if __name__ == "__main__":
    # Parse PDBBind v2020 refined-set
    pdbb = parse_refined_set("refined-set-aligned", 2020)

    # Collect molecular weigth
    for idx, row in pdbb.iterrows():
        if not os.path.exists("refined-set-aligned/%s_%s_ligand.pdb" % (row.uniprot, row.pdb)):
            continue
        mw = extract_mw("refined-set-aligned/%s_%s_ligand.pdb" % (row.uniprot, row.pdb))
        pdbb.at[idx, "molecular_weigth"] = mw
    
    # Output dataset
    pdbb.to_csv("refined-set-v2020.csv", index=False)

    # Align UniProts by smallest ligand
    for uniprot, group in pdbb.groupby("uniprot"):
        if uniprot == "------" or len(group) == 1 or uniprot == '000000':
            continue
        pm.reinitialize()

        if True:
            # Get the smallest ligand
            try:
                min_pdb = list(
                    group.pdb[group.molecular_weigth == min(group.molecular_weigth)]
                )[0]
            except:
                continue
            for pdb in group.pdb:
                try:
                    pm.load("refined-set-aligned/%s_%s_protein.pdb" % (uniprot, pdb))
                except:
                    continue
            try:
                pm.extra_fit(
                    "*_protein",
                    "%s_%s_protein" % (uniprot, min_pdb),
                    method="align",
                    cycles=5,
                    cutoff=2.0,
                )

                for pdb in group.pdb:
                    pm.load("refined-set-aligned/%s_%s_ligand.pdb" % (uniprot, pdb))
                    pm.matrix_copy(
                        "%s_%s_protein" % (uniprot, min_pdb),
                        "%s_ligand"  % (uniprot, min_pdb)
                    )
            except:
                continue
        # Save all objects to the output directory
        pm.alter("*_ligand", 'resn="LIG"')
        for obj in pm.get_object_list():
            pm.save("refined-set-aligned/%s_%s.pdb" % (uniprot, obj), obj)