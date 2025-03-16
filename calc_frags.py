import pandas as pd
import numpy as np
import seaborn as sb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from matplotlib import pyplot as plt
from openbabel import pybel as pb
from collections import namedtuple
from xdrugpy.hotspots import load_ftmap, fo, dce, ho
from pymol import cmd as pm
from glob import glob
from collections import defaultdict
from scipy.stats import kruskal, mannwhitneyu, kendalltau
from pprint import pprint


FRAG_NHA_THRESHOLD = 3
NHA_THRESHOLD = 3
OVERLAP_THRESHOLD = 0.7
FO1_THRESHOLD = 0.8


def get_fp(mol):
    mol = Chem.rdmolops.AddHs(mol)
    fpgen = AllChem.GetRDKitFPGenerator(useHs=False)
    fp = fpgen.GetFingerprint(mol)
    return fp


rs = pd.read_csv("refined-set-v2020.csv")

Node = namedtuple(
    'Node',
    'uniprot pdb nha mw pki le bei fp'
)
Edge = namedtuple(
    'Edge',
    'pdb1 pdb2'
)


def find_proteins():
    prots = rs.sort_values('uniprot')
    uniprot = prots.iloc[0].uniprot
    last_group = []
    for i, row in prots.iterrows():
        if row.uniprot == uniprot:
            last_group.append(row)
        else:
            yield uniprot, last_group
            uniprot = row.uniprot
            last_group = []

def overlap(fp1, fp2):
    assert len(fp1) == len(fp2)
    return sum(fp1 & fp2) / min(sum(fp1), sum(fp2))



count_pdb = set()
count_uniprot = set()
signs = defaultdict(list)
boxplot = defaultdict(list)
hist = defaultdict(list)
ligands = []
nodes = []
for uniprot, group in find_proteins():

    if uniprot == '000000' or len(group) < 5 or uniprot in []:
        continue
    
    # if uniprot not in ['A7YT55', 'O96935', 'P00517', 'P00730', 'P00749', 'P00760', 'P00760', 'P00760', 'P00760', 'P00760', 'P00800', 'P00800', 'P00800', 'P00800', 'P00800', 'P00918', 'P00918', 'P00918', 'P00918', 'P00918', 'P00918', 'P00918', 'P00918', 'P00918', 'P02754', 'P03366', 'P03366', 'P03367', 'P04058', 'P04585', 'P07900', 'P12821', 'P12821', 'P15090', 'P17931', 'P17931', 'P17931', 'P18031', 'P22734', 'P24941', 'P29317', 'P29317', 'P37231', 'P37231', 'P39900', 'P39900', 'P39900', 'P39900', 'P49773', 'P56817', 'P56817', 'P61823', 'P61823', 'P61823', 'Q07820', 'Q08638', 'Q10714', 'Q5SLD7', 'Q9Y233']:
    #     continue
    # if uniprot not in ["O96935", 'A7YT55', 'P03367', 'P00517']:
    #     continue

    # if uniprot != "O43570":
    #     continue
    u_nodes = []
    visited = set()
    for row in group:

        # skip repeated ligands
        if row.ligand in visited:
            continue   
        visited.add(row.ligand)
        try:
        # load molecule
            mol = Chem.MolFromPDBFile(f'refined-set-aligned/{uniprot}_{row.pdb}_ligand.pdb', sanitize=False)
            # mol = Chem.MolFromPDBFile(lig_fname, sanitize=False)
        except:
            continue
        # create node
        nha = len(mol.GetAtoms())
        pki = row.pki
        le = pki / nha
        bei = pki / Descriptors.ExactMolWt(mol)
        fp = get_fp(mol)

        node = Node(
            uniprot=uniprot,
            pdb=row.pdb.upper(),
            nha=nha,
            mw=Descriptors.ExactMolWt(mol),
            pki=pki,
            le=le,
            bei=bei,
            fp=fp,
        )
        print(f"{node.pdb=}\t{node.pki=}")
        u_nodes.append(node)
        nodes.append(node)

    # Load a representative FTMap structure
    print(uniprot, 'Loading FTMap')
    pm.reinitialize()
    rep = glob(f'refined-set-aligned/{uniprot}_????_atlas.pdb')
    if len(rep) == 0:
        continue
    rep = rep[0]
    ref_pdb = rep.split('/')[-1].split('_')[1]
    pm.load(f'refined-set-aligned/{uniprot}_{ref_pdb}_ligand.pdb')
    ftmap = load_ftmap(rep)
    found_hs = False
    for hs in ftmap.kozakov2015:
        if fo(hs.selection, f'{uniprot}_{ref_pdb}_ligand', verbose=False) > 0:
            found_hs = True
            break
    found_cs = False
    for cs in ftmap.clusters:
        if fo(cs.selection, f'{uniprot}_{ref_pdb}_ligand', verbose=False) > 0:
            found_cs = True
            break
    if not found_hs or not found_cs:
        continue
    print(uniprot, "Compute overalap")
    fo_d = {}
    for node in u_nodes:
        try:
            lig_obj = f'{uniprot}_{node.pdb}_ligand'
            pm.load(f'refined-set-aligned/{uniprot}_{node.pdb.lower()}_ligand.pdb')
            if not (found_hs and found_cs):
                continue
            fo1_hs = fo(hs.selection, lig_obj, verbose=False)
            fo2_hs = fo(lig_obj, hs.selection, verbose=False)
            fo1_cs = fo(cs.selection, lig_obj, verbose=False)
            fo2_cs = fo(lig_obj, cs.selection, verbose=False)
            fo21_hs = fo2_hs - fo1_hs
            dce_hs = dce(lig_obj, hs.selection, verbose=False)
            dce_cs = dce(lig_obj, cs.selection, verbose=False)
            ligands.append({
                'type': 'hs',
                'uniprot': uniprot,
                'pdb': node.pdb,
                'fo1_hs': fo1_hs,
                'fo2_hs': fo2_hs,
                'fo1_cs': fo1_cs,
                'fo2_cs': fo2_cs,
                'fo21_hs': fo21_hs,
                'dce_hs': dce_hs,
                'dce_cs': dce_cs,
                'pki': node.pki,
                'le': node.le,
                'bei': node.bei,
                'nha': node.nha,
                'mw': node.mw,
            })
            fo_d[node.pdb] = {
                'fo1_hs': fo1_hs,
                'fo2_hs': fo2_hs,
                'fo1_cs': fo1_cs,
                'fo2_cs': fo2_cs,
                'fo21_hs': fo21_hs,
                'dce_hs': dce_hs,
                'dce_cs': dce_cs,
            }
            pm.delete(lig_obj)

        except:
            raise

    # Find similar ligands based on overlap similarity
    for i1, n1 in enumerate(u_nodes):
        for i2, n2 in enumerate(u_nodes):
            if i1 >= i2:
                continue
            fp1 = n1.fp
            fp2 = n2.fp
            ovr = overlap(fp1, fp2)
            nha_diff = abs(n1.nha - n2.nha)
            if ovr > OVERLAP_THRESHOLD and nha_diff > 0:# and nha_diff > NHA_THRESHOLD:
                if n1.pdb not in fo_d or n2.pdb not in fo_d:
                    continue
                else:
                    count_pdb.update([n1.pdb, n2.pdb])
                    count_uniprot.add(uniprot)
                for fo_key in ['fo1_hs', 'fo2_hs', 'fo1_cs', 'fo2_cs']:
                    for a_key in ['pki', 'le']:
                        fo_value1 = fo_d[n1.pdb][fo_key]
                        fo_value2 = fo_d[n2.pdb][fo_key]
                        a_value1 = getattr(n1, a_key)
                        a_value2 = getattr(n2, a_key)
                        delta = (fo_value1 - fo_value2) * (a_value1 - a_value2)
                        signs[(a_key, fo_key)].append(delta)
                        

    # Calculate the PKI variance for the maximal affinity partition
    low_fo1_pki = []
    top_fo1_pki = []
    for node in u_nodes:
        if node.pdb not in fo_d:
            continue
        if fo_d[node.pdb]['fo1_hs'] < FO1_THRESHOLD:
            low_fo1_pki.append(node.pki)
            hist['low_fo1_pki'].append(node.pki)
        else:
            top_fo1_pki.append(node.pki)
            hist['top_fo1_pki'].append(node.pki)
    if low_fo1_pki:
        boxplot['low_fo1_pki'].append(np.var(low_fo1_pki))
    if top_fo1_pki:
        boxplot['top_fo1_pki'].append(np.var(top_fo1_pki))




import seaborn as sb


pd.DataFrame(ligands).to_csv('ligands.csv', index=False)



count_uniprot = len(count_uniprot)
count_pdb = len(count_pdb)
print(f'{count_uniprot=}, {count_pdb=}')






fig, ax = plt.subplots(layout="constrained")

correlations = {
    '+1':0,
    '-1':0,
    '0': 0
}
activities = []
for ovr in [
    'fo1_hs', 'fo2_hs', 'fo21_hs',
    'fo2_cs', 'fo2_cs', 'fo21_cs',
    'dce_hs', 'dce_cs'
]:
    for act in ['pki', 'le', 'bei']:
        corr = sum(np.array(signs[(act, ovr)]))
        if corr > 0:
            correlations['+1'] += 1
        elif corr < 0:
            correlations['-1'] += 1
        else:
            correlations['0'] += 1
        activities.append('%s/%s' % (ovr, act))

pd.DataFrame(correlations, index=activities).reset_index().rename(
    columns={
        "+1": "positive",
        "-1": "negative",
        "0": "zero"
    }
).to_csv('cors.csv')






print(mannwhitneyu(
    boxplot['low_fo1_pki'],
    boxplot['top_fo1_pki'],
    alternative='greater'
))
pprint({
    'Median(Var(low_fo1))': np.median(boxplot['low_fo1_pki']),
    'Median(Var(top_fo1))': np.median(boxplot['top_fo1_pki']),
})


with open('max_pki.txt', 'w') as f:
    f.write(' '.join(map(str, boxplot['low_fo1_pki'])))
    f.write('\n')
    f.write(' '.join(map(str, boxplot['top_fo1_pki'])))
    f.write('\n')
    f.write(' '.join(map(str, hist['low_fo1_pki'])))
    f.write('\n')
    f.write(' '.join(map(str, hist['top_fo1_pki'])))
