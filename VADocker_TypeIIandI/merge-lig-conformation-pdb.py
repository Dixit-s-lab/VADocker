#!/usr/bin/env python3
"""
merge_ligions.py

Merges two ligand conformations from two protein-ligand complex PDB files into a 
single PDB complex, formatting the ligand atoms with 'A' and 'B' suffixes, 
0.50 occupancy, sequential serial numbers, and preserving full CONECT connectivity records.
"""

import argparse
import sys
from pathlib import Path

STANDARD_PROTEIN_RES = {
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
    'ASX', 'GLX', 'SEC', 'PYL', 'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'T', 'U'
}
SOLVENT_AND_IONS = {
    'HOH', 'WAT', 'TIP3', 'H2O', 'NA', 'CL', 'MG', 'ZN', 'CA', 'K', 'FE', 'SO4', 'PO4'
}

def parse_pdb_line(line: str):
    """Parses a PDB ATOM or HETATM line into a structured dictionary."""
    if not line.startswith(('ATOM  ', 'HETATM')):
        return None
    
    record = line[0:6].strip()
    try:
        serial = int(line[6:11].strip())
    except ValueError:
        serial = 0
    atom_name = line[12:16].strip()
    alt_loc = line[16:17].strip()
    res_name = line[17:20].strip().upper()
    chain = line[21:22].strip()
    try:
        res_num = int(line[22:26].strip())
    except ValueError:
        res_num = 0
    ins_code = line[26:27].strip()
    
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        x, y, z = 0.0, 0.0, 0.0
        
    try:
        occupancy = float(line[54:60])
    except ValueError:
        occupancy = 1.00
        
    try:
        temp_factor = float(line[60:66])
    except ValueError:
        temp_factor = 0.00
        
    element = line[76:78].strip().upper()
    if not element and atom_name:
        alpha = "".join([c for c in atom_name if c.isalpha()]).upper()
        element = alpha[0] if alpha else "C"
        
    return {
        "record": record,
        "serial": serial,
        "atom_name": atom_name,
        "alt_loc": alt_loc,
        "res_name": res_name,
        "chain": chain,
        "res_num": res_num,
        "ins_code": ins_code,
        "x": x,
        "y": y,
        "z": z,
        "occupancy": occupancy,
        "temp_factor": temp_factor,
        "element": element,
        "raw": line
    }

def format_atom_line(atom_dict, new_serial):
    """Formats an atom dictionary back into a standard fixed-width PDB line."""
    record = atom_dict["record"]
    atom_name = atom_dict["atom_name"]
    alt_loc = atom_dict["alt_loc"]
    res_name = atom_dict["res_name"]
    chain = atom_dict["chain"]
    res_num = str(atom_dict["res_num"])
    ins_code = atom_dict["ins_code"]
    x = atom_dict["x"]
    y = atom_dict["y"]
    z = atom_dict["z"]
    occupancy = atom_dict["occupancy"]
    temp_factor = atom_dict["temp_factor"]
    element = atom_dict["element"]
    
    return f"{record:<6s}{new_serial:5d} {atom_name:<4s}{alt_loc:1s}{res_name:3s} {chain:1s}{res_num:>4s}{ins_code:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}\n"

def detect_ligand_residue(atoms, ligand_resname=None):
    """Automatically detects the ligand residue name if not provided."""
    if ligand_resname:
        return ligand_resname.upper()
    
    res_counts = {}
    for a in atoms:
        r = a["res_name"]
        if r not in STANDARD_PROTEIN_RES and r not in SOLVENT_AND_IONS and a["record"] == "HETATM":
            res_counts[r] = res_counts.get(r, 0) + 1
            
    if not res_counts:
        print("Error: Could not automatically detect ligand residue. Please specify using --ligand-resname.", file=sys.stderr)
        sys.exit(1)
        
    return max(res_counts, key=res_counts.get)

def parse_conect_records(file_path, old_to_new_serial):
    """Parses CONECT records from a PDB file and remaps old serial numbers to new ones."""
    conect_map = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('CONECT'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        orig_src = int(parts[1])
                        if orig_src in old_to_new_serial:
                            new_src = old_to_new_serial[orig_src]
                            new_targets = []
                            for target_str in parts[2:]:
                                orig_tgt = int(target_str)
                                if orig_tgt in old_to_new_serial:
                                    new_targets.append(old_to_new_serial[orig_tgt])
                            if new_src not in conect_map:
                                conect_map[new_src] = set()
                            conect_map[new_src].update(new_targets)
                    except ValueError:
                        continue
    return conect_map

def main():
    parser = argparse.ArgumentParser(description="Merge two ligand conformations into a single PDB complex with CONECT records.")
    parser.add_argument("--complex1", required=True, type=Path, help="Path to first protein-ligand complex PDB")
    parser.add_argument("--complex2", required=True, type=Path, help="Path to second protein-ligand complex PDB")
    parser.add_argument("--output", required=True, type=Path, help="Path to output merged PDB complex")
    parser.add_argument("--ligand-resname", type=str, default=None, help="Ligand residue name (e.g., DMZ). Auto-detected if omitted.")
    
    args = parser.parse_args()
    
    if not args.complex1.is_file():
        print(f"Error: Complex 1 file not found: {args.complex1}", file=sys.stderr)
        sys.exit(1)
    if not args.complex2.is_file():
        print(f"Error: Complex 2 file not found: {args.complex2}", file=sys.stderr)
        sys.exit(1)
        
    # Read atoms from both files
    atoms1 = []
    with args.complex1.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            atom = parse_pdb_line(line)
            if atom:
                atoms1.append(atom)
                
    atoms2 = []
    with args.complex2.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            atom = parse_pdb_line(line)
            if atom:
                atoms2.append(atom)
                
    lig_res = detect_ligand_residue(atoms1, args.ligand_resname)
    print(f"-> Detected Ligand Residue Name: {lig_res}")
    
    protein_atoms = [a for a in atoms1 if a["res_name"] != lig_res]
    ligand_atoms_1 = [a for a in atoms1 if a["res_name"] == lig_res]
    ligand_atoms_2 = [a for a in atoms2 if a["res_name"] == lig_res]
    
    if not ligand_atoms_1 or not ligand_atoms_2:
        print(f"Error: Ligand '{lig_res}' not found in one of the complex files.", file=sys.stderr)
        sys.exit(1)
        
    # Build mapping from original serials to new sequential serials
    old_to_new_1 = {}
    old_to_new_2 = {}
    
    current_serial = 1
    
    # Protein mapping
    for a in protein_atoms:
        old_to_new_1[a["serial"]] = current_serial
        current_serial += 1
        
    # Ligand 1 (Conformation A) mapping
    for a in ligand_atoms_1:
        old_to_new_1[a["serial"]] = current_serial
        current_serial += 1
        
    # Ligand 2 (Conformation B) mapping
    # Reset serial tracking for complex 2 protein/ligand mapping if needed, 
    # but here protein is shared from complex 1.
    temp_serial_2 = 1
    for a in [a for a in atoms2 if a["res_name"] != lig_res]:
        temp_serial_2 += 1
    for a in ligand_atoms_2:
        old_to_new_2[a["serial"]] = current_serial
        current_serial += 1

    # Process Conformation A (Suffix 'A', occupancy 0.50)
    processed_ligand_1 = []
    for a in ligand_atoms_1:
        new_atom = dict(a)
        base_name = a["atom_name"].rstrip("AB")
        new_atom["atom_name"] = (base_name + "A")[:4]
        new_atom["occupancy"] = 0.50
        processed_ligand_1.append(new_atom)
        
    # Process Conformation B (Suffix 'B', occupancy 0.50)
    processed_ligand_2 = []
    for a in ligand_atoms_2:
        new_atom = dict(a)
        base_name = a["atom_name"].rstrip("AB")
        new_atom["atom_name"] = (base_name + "B")[:4]
        new_atom["occupancy"] = 0.50
        processed_ligand_2.append(new_atom)
        
    all_atoms = protein_atoms + processed_ligand_1 + processed_ligand_2
    
    # Parse and remap CONECT connectivity records
    conect_map_1 = parse_conect_records(args.complex1, old_to_new_1)
    conect_map_2 = parse_conect_records(args.complex2, old_to_new_2)
    
    # Merge connect maps
    combined_conect = {}
    for src, targets in conect_map_1.items():
        if src not in combined_conect:
            combined_conect[src] = set()
        combined_conect[src].update(targets)
    for src, targets in conect_map_2.items():
        if src not in combined_conect:
            combined_conect[src] = set()
        combined_conect[src].update(targets)

    # Write out merged PDB
    output_lines = []
    current_serial = 1
    for a in all_atoms:
        output_lines.append(format_atom_line(a, current_serial))
        current_serial += 1
        
    # Append CONECT records
    for src in sorted(combined_conect.keys()):
        targets = sorted(list(combined_conect[src]))
        # PDB format allows up to 4 connection targets per CONECT line
        for i in range(0, len(targets), 4):
            chunk = targets[i:i+4]
            targets_str = "".join(f"{t:5d}" for t in chunk)
            output_lines.append(f"CONECT{src:5d}{targets_str}\n")
            
    output_lines.append("END\n")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(output_lines), encoding='utf-8')
    print(f"-> Successfully merged complexes with CONECT records into: {args.output}")
    print(f"   Total atoms written: {len(all_atoms)}")

if __name__ == "__main__":
    main()