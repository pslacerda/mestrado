from os.path import dirname, basename, splitext
from subprocess import check_call
from pymol import cmd as pm
from glob import glob


for pse in glob("kellenberger/*.pse"):
    pm.reinitialize()
    uniprot = basename(pse)[:6]

    # find apo
    print("Loading", uniprot)
    pm.load(pse)
    apo_obj = pm.get_object_list("apo")[0]
    apo_fname = f'kellenberger/{uniprot}_apo_{apo_obj}.pdb'
    pm.save(apo_fname, apo_obj)
    
    # run atlas
    prefix = apo_fname[:-4]
    if glob(f"{prefix}_atlas*"):
        continue
    print("Running", prefix)
    check_call(
        f"run_atlas --prefix {prefix}_atlas {apo_fname}",
        shell=True
    )

    # find holo ligands
    objs = pm.get_object_list("holo & enabled")
    for obj in objs:
        lig_fname = f'kellenberger/{uniprot}_{obj}.mol2'
        pm.save(lig_fname, f'!polymer & %{obj}')

