import numpy as np
from os.path import exists, basename, splitext
from pymol import cmd as pm
from glob import glob
from openbabel import pybel as pb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


pm.set("pdb_conect_all", 1)


def overlap(fp1, fp2):
    assert len(fp1) == len(fp2)
    return sum(fp1 & fp2) / min(sum(fp1), sum(fp2))


for pse in glob("kellenberger/*.pse"):
    pm.reinitialize()
    pm.load(pse)

    key = basename(splitext(pse)[0])
    uniprot = basename(key.split('_')[0])

    # if key != "P07900_series3":
    #     continue

    print(key)
    visited_ligs = set()
    mols = {}
    objs = pm.get_object_list("holo & enabled")
    for obj in objs:
        lig_sel = f'!polymer & !resn HOH & !inorganic & %{obj}'

        # check duplicate ligands
        resns = set(
            (a.resn, a.resi, a.chain)
            for a in pm.get_model(lig_sel).atom
        )
        if len(set(resns)) != 1:
            print("Bad: multiple ligands:", key, obj)
            continue
        resn = list(resns)[0]
        if resn in visited_ligs:
            continue
        visited_ligs.add(resn)
        
        # compute fingerprints
        lig_fname = f'kellenberger/{uniprot}_{obj}_lig.mol2'
        pm.save(lig_fname, lig_sel)
        mol = Chem.MolFromMol2File(lig_fname, sanitize=False)
        mol = Chem.rdmolops.AddHs(mol)
        fpgen = AllChem.GetRDKitFPGenerator(useHs=False)
        fp = fpgen.GetFingerprint(mol)
        mw = Descriptors.ExactMolWt(mol)
        mols[obj] = (fp, mw)

    # find smaller ligand as fragment
    min_mw = float('inf')
    min_pdb = None
    min_fp = None
    for pdb, (fp, mw) in mols.items():
        if mw < min_mw:
            min_mw = mw
            min_pdb = pdb
            min_fp = fp
    
    # compute overlap
    for idx, (pdb, (fp, _)) in enumerate(mols.items()):
        ovr = overlap(min_fp, fp)
        print(key, min_pdb, pdb, "%.2f" % ovr)
    print()
    