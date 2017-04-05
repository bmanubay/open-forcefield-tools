from openeye.oechem import *
from openeye.oedepict import *

def DrawCellNumber(cell, idx):
    font = OEFont()
    font.SetAlignment(OEAlignment_Center)
    yoffset = font.GetSize()
    xoffset = font.GetSize()
    cell.DrawText(OE2DPoint(xoffset, yoffset), str(idx), font)

smiles = ["C","CC","CCC","COC","CCOC","C1COC1","CO","CCO","CCCO","CC(C)O","O"] 

image = OEImage(800, 800)

rows, cols = 6, 2
grid = OEImageGrid(image, rows, cols)

opts = OE2DMolDisplayOptions(grid.GetCellWidth(), grid.GetCellHeight(), OEScale_AutoScale)

for idx, cell in zip(range(0, len(smiles)), grid.GetCells()):
    mol = OEGraphMol()
    OESmilesToMol(mol, smiles[idx])
    OEPrepareDepiction(mol)

    disp = OE2DMolDisplay(mol, opts)
    OERenderMolecule(cell, disp)

    OEDrawBorder(cell, OEBlackPen)
    DrawCellNumber(cell, idx + 1)

OEWriteImage("AlkEthOH_bayes1.png", image)
