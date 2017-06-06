import glob
import pandas as pd
from openeye.oechem import *
from collections import OrderedDict
import pdb

molfiles = glob.glob('*.mol2')

AlkEthOH_ID = []
SMILES_strings = []
InChI_keys = []
Molecular_formula = []

for moldex,j in enumerate(molfiles):
    #if moldex>10:
        #pdb.set_trace()
    mol = OEGraphMol()
    ifs = oemolistream(j)
    ofs = oemolistream(j)
    flavorin = OEIFlavor_Generic_Default | OEIFlavor_MOL2_Default | OEIFlavor_MOL2_Forcefield
    flavorout = OEOFlavor_INCHIKEY_Stereo
    ifs.SetFlavor(OEFormat_MOL2, flavorin)
    ofs.SetFlavor(OEFormat_INCHIKEY, flavorout)
    OEReadMolecule(ifs, mol)
    OETriposAtomNames(mol)
    
    smi = OEMolToSmiles(mol)
    InChIkey = OECreateInChIKey(mol)
    form = OEMolecularFormula(mol)
    
    AlkEthOH_ID.append(j.split('.')[0])
    SMILES_strings.append(smi)
    InChI_keys.append(InChIkey)
    if "S" not in InChIkey.split('-')[1]:
        pdb.set_trace()
    Molecular_formula.append(form)

df = pd.DataFrame({
                   'InChI_keys':InChI_keys,
                   'SMILES_strings':SMILES_strings,
                   'Molecular_formulas':Molecular_formula,
                   'AlkEthOH_ID':AlkEthOH_ID
                  })
df = df[['InChI_keys','SMILES_strings','Molecular_formulas','AlkEthOH_ID']]

df.to_csv('AlkEthOH_data_set_identifiers.txt',sep='\t')
    


