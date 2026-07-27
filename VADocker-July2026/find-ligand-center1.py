#!/usr/bin/env python3
import sys
from pathlib import Path

def calculate_ligand_center(pdb_path_str):
    pdb_path = Path(pdb_path_str)
    
    if not pdb_path.is_file():
        print(f"❌ Error: Could not find file '{pdb_path_str}'")
        sys.exit(1)
        
    x_coords = []
    y_coords = []
    z_coords = []
    atom_count = 0
    
    # Parse lines matching strict PDB specifications
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                # PDB Column allocations: X (30-38), Y (38-46), Z (46-54)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                atom_count += 1
            except (ValueError, IndexError):
                # Safely ignore malformed lines or comment text rows
                continue
                
    if atom_count == 0:
        print(f"❌ Error: No valid ATOM or HETATM coordinate records found inside '{pdb_path_str}'")
        sys.exit(1)
        
    # Compute the true spatial geometric center (mean values)
    cx = sum(x_coords) / atom_count
    cy = sum(y_coords) / atom_count
    cz = sum(z_coords) / atom_count
    
    print("=" * 60)
    print(f"🎯 COORDINATE CENTER MATRIX FOR: {pdb_path.name}")
    print(f"   Parsed Atoms: {atom_count}")
    print("=" * 60)
    print(f"X: {cx:.3f}")
    print(f"Y: {cy:.3f}")
    print(f"Z: {cz:.3f}")
    print("=" * 60)
    print(f"\n📋 Assembled VADocker String Parameters fragment:")
    print(f"--center {cx:.3f} {cy:.3f} {cz:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    # Allows passing the filename directly as an argument or falls back to prompt
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = input("[?] Enter name/path of Ligand PDB (e.g., EST-H.pdb): ").strip()
        
    calculate_ligand_center(target_file)