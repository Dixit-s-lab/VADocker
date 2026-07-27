#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from core_engine_empirical import read_pdb, execute_single_docking

def pool_worker(args_tuple):
    return execute_single_docking(*args_tuple)

def main():
    parser = argparse.ArgumentParser(description="VADocker High-Performance Parallel Docking Driver Interface")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ligands", type=Path, help="Directory folder containing all ligand structure files for batch docking analysis.")
    group.add_argument("--ligand", type=Path, help="Explicit singular ligand target configuration file path (.pdb, .sdf, or .mol2).")
    
    parser.add_argument("--receptor", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--translation-step", type=float, default=1.5)
    parser.add_argument("--rotation-step", type=float, default=20.0)
    parser.add_argument("--top-poses", type=int, default=10)
    parser.add_argument("--scoring", choices=["empirical", "amber"], default="empirical")
    parser.add_argument("--cores", type=int, default=cpu_count())
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--smiles", type=str, default=None, help="Explicit SMILES string to enforce correct hybridization template configuration.")

    args = parser.parse_args()
    
    if not args.receptor.is_file():
        print(f"Error: Receptor configuration layout target missing at {args.receptor}")
        sys.exit(1)
        
    receptor_atoms = read_pdb(args.receptor)
    ref_atoms = read_pdb(args.reference) if args.reference and args.reference.is_file() else None
    
    if args.ligand:
        if not args.ligand.is_file():
            print(f"Error: Specified target ligand file does not exist: {args.ligand}")
            sys.exit(1)
        ligand_files = [args.ligand]
    else:
        if not args.ligands.is_dir():
            print(f"Error: Specified batch directory pathway does not exist: {args.ligands}")
            sys.exit(1)
        ligand_files = sorted(f for f in args.ligands.iterdir() if f.is_file() and f.suffix.lower() in {".pdb", ".mol2", ".sdf"})

    if not ligand_files:
        print("Error: Target validation streams returned 0 executable ligand structure maps.")
        sys.exit(1)
        
    print("="*60)
    print(f"🧬 LATTICE SEARCH RUN STARTED | Engine Track: Parallel CPU Mode ({args.cores} cores)")
    print(f" -> Scoring Routine Selected: {args.scoring.upper()} Potential Function")
    print(f" -> Queue Status: Processing {len(ligand_files)} dynamic input ligand arrays.")
    if ref_atoms:
        print(f" -> Reference Ligand Status: Loaded {len(ref_atoms)} reference atoms cleanly.")
    else:
        print(" -> Reference Ligand Status: None provided or file path unreadable.")
    print("="*60)
    
    tasks = []
    for lf in ligand_files:
        # STRICT POSITIONAL MAPPING: Ensure this order aligns perfectly with the backend signature
        tasks.append((
            receptor_atoms, lf, args.out, tuple(args.center), 
            args.size, args.translation_step, args.rotation_step, 
            args.scoring, args.top_poses, ref_atoms, args.smiles
        ))
        
    with Pool(processes=args.cores) as pool:
        for output_status in pool.imap_unordered(pool_worker, tasks):
            print(f" [✔] Status Update: {output_status}")
            
    print("\n🎉 Task Processing arrays executed successfully.")

if __name__ == "__main__":
    main()