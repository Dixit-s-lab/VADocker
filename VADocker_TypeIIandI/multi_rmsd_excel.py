#!/usr/bin/env python3
"""Interactive unaligned ligand RMSD; RDKit infers missing PDB connectivity."""
import glob
import os
import numpy as np
from rdkit import Chem
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def read_ligand(path):
    # proximityBonding reconstructs bonds when CONECT records are absent.
    mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumConformers() != 1:
        raise ValueError('Could not read a single PDB ligand conformer')
    mol = Chem.RemoveHs(mol, sanitize=False)
    if not mol.GetNumAtoms():
        raise ValueError('No heavy atoms found')
    return mol


def rmsd(ref, pose):
    if ref.GetNumAtoms() != pose.GetNumAtoms():
        raise ValueError(f'Heavy atom mismatch: {ref.GetNumAtoms()} vs {pose.GetNumAtoms()}')
    # Use graph connectivity and element identity. No ligand alignment is performed.
    # RDKit substructure matching checks bond types, which can differ in PDB inference;
    # use generic bonds to compare topology without claiming bond-order equivalence.
    def generic(m):
        rw = Chem.RWMol(m)
        for bond in rw.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        for atom in rw.GetAtoms():
            atom.SetIsAromatic(False)
            atom.SetFormalCharge(0)
        return rw.GetMol()
    r, p = generic(ref), generic(pose)
    if sorted(a.GetAtomicNum() for a in r.GetAtoms()) != sorted(a.GetAtomicNum() for a in p.GetAtoms()):
        raise ValueError('Heavy atom element counts differ')
    matches = p.GetSubstructMatches(r, uniquify=False, maxMatches=1000000)
    if not matches:
        raise ValueError('No full connectivity match; inferred bonds may differ (check PDB/SDF)')
    rc = ref.GetConformer().GetPositions()
    pc = pose.GetConformer().GetPositions()
    best = min(float(np.sqrt(np.mean(np.sum((rc-pc[list(match)])**2, axis=1)))) for match in matches)
    return best, len(matches)


def main():
    print('\n========== BIOVIA LIGAND RMSD ==========')
    reference = input('Enter reference ligand PDB filename: ').strip().strip('"\'')
    pattern = input('Enter docked conformer pattern (e.g. 3UA1_group*_pose*.pdb): ').strip().strip('"\'')
    if not os.path.isfile(reference):
        print('Reference file not found:', reference)
        return
    files = [f for f in sorted(glob.glob(pattern)) if os.path.abspath(f) != os.path.abspath(reference)]
    if not files:
        print('No conformer files found for:', pattern)
        return
    try:
        ref = read_ligand(reference)
    except Exception as e:
        print('Reference ERROR:', e)
        return
    print(f'\nReference: {reference}\nReference heavy atoms: {ref.GetNumAtoms()}\nConformers found: {len(files)}')
    print(f'\n{"Conformer":40} {"RMSD (Å)":>10}  Status')
    print('-'*75)
    results = []
    errors = []
    for path in files:
        try:
            value, n = rmsd(ref, read_ligand(path))
            results.append((path, value))
            print(f'{path:40} {value:10.3f}  {"Within 2 Å" if value <= 2 else "Above 2 Å"}')
        except Exception as e:
            errors.append((path, str(e)))
            print(f'{path:40} ERROR  {e}')
    print('-'*75)
    if results:
        name, value = min(results, key=lambda item: item[1])
        print(f'Best conformer: {name}\nBest RMSD: {value:.3f} Å')
    else:
        print('No RMSD values calculated successfully.')
    # Excel workbook: include successful poses and failed poses for auditing.
    output = os.path.join(os.path.dirname(os.path.abspath(reference)),
                          os.path.splitext(os.path.basename(reference))[0] + '_RMSD_results.xlsx')
    wb = Workbook()
    ws = wb.active
    ws.title = 'RMSD Results'
    ws.append(['BIOVIA Docking RMSD Results'])
    ws.append(['Reference ligand', os.path.basename(reference)])
    ws.append(['Reference heavy atoms', ref.GetNumAtoms()])
    ws.append(['Total conformers', len(files)])
    ws.append(['Successfully calculated', len(results)])
    ws.append(['Failed', len(errors)])
    ws.append([])
    ws.append(['Rank', 'Conformer', 'RMSD (Å)', 'Status'])
    for rank, (path, value) in enumerate(sorted(results, key=lambda x: x[1]), 1):
        ws.append([rank, os.path.basename(path), value,
                   'Within 2 Å' if value <= 2.0 else 'Above 2 Å'])
        ws.cell(ws.max_row, 3).number_format = '0.0000'
    ws.freeze_panes = 'C9'
    ws.auto_filter.ref = f'A8:D{max(8, ws.max_row)}'
    ws.column_dimensions['A'].width = 12
    ws.column_dimensions['B'].width = 42
    ws.column_dimensions['C'].width = 18
    ws.column_dimensions['D'].width = 20
    for cell in ws[8]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='23466B')
        cell.alignment = Alignment(horizontal='center')
    ws['A1'].font = Font(size=15, bold=True, color='23466B')
    for row in range(9, ws.max_row + 1):
        if row % 2 == 0:
            for cell in ws[row]:
                cell.fill = PatternFill('solid', fgColor='EFF4F9')
    if errors:
        err = wb.create_sheet('Errors')
        err.append(['Conformer', 'Error'])
        for path, message in errors:
            err.append([os.path.basename(path), message])
        err.column_dimensions['A'].width = 42
        err.column_dimensions['B'].width = 85
        for cell in err[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='23466B')
    try:
        wb.save(output)
        print(f'Excel results saved: {output}')
    except Exception as e:
        print(f'Could not save Excel results: {e}')
    print('Note: bonds inferred from distances; check connectivity and coordinate frame before reporting.')

if __name__ == '__main__':
    main()
