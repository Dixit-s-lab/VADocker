#!/usr/bin/env python3
import sys
import os
import argparse
import csv
import math
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from core_engine_empirical import read_pdb, execute_single_docking

def min_max_scale(val, val_list):
    min_v, max_v = min(val_list), max(val_list)
    if max_v == min_v:
        return 0.0
    return (val - min_v) / (max_v - min_v)

def apply_composite_scoring(rows, has_reference=False):
    """
    Evaluates ALL generated poses using an equal-weight composite function.
    - With Reference RMSD (>0): 5 parameters @ 20% weight each (0.20)
    - Without Reference RMSD (=0): 4 parameters @ 25% weight each (0.25)
    """
    if not rows:
        return rows

    # Extract full population arrays for normalization bounds
    raw_scores = [float(r.get("raw_score", r.get("score", 0.0))) for r in rows]
    clashes = [float(r.get("clashes", 0.0)) for r in rows]
    contacts = [float(r.get("contacts", 0.0)) for r in rows]
    hbonds = [float(r.get("hbonds", 0.0)) for r in rows]
    rmsds = [float(r.get("rmsd_to_reference", 0.0)) for r in rows]

    valid_rmsd = has_reference and any(r > 0.0 for r in rmsds)

    # Dynamic Equal Weighting Assignment
    if valid_rmsd:
        w = 0.20
        w_score = w_clashes = w_contacts = w_hbonds = w_rmsd = w
    else:
        w = 0.25
        w_score = w_clashes = w_contacts = w_hbonds = w
        w_rmsd = 0.0

    for i, r in enumerate(rows):
        n_raw = min_max_scale(raw_scores[i], raw_scores)
        n_clashes = min_max_scale(clashes[i], clashes)
        n_contacts = min_max_scale(contacts[i], contacts)
        n_hbonds = 1.0 - min_max_scale(hbonds[i], hbonds) # Invert: more H-bonds = closer to 0 penalty
        n_rmsd = min_max_scale(rmsds[i], rmsds) if valid_rmsd else 0.0

        r["raw_score"] = f"{raw_scores[i]:.4f}"

        # Calculate Master Composite Score (Lower = Better Overall Structure)
        composite_val = (
            w_score * n_raw +
            w_clashes * n_clashes +
            w_contacts * n_contacts +
            w_hbonds * n_hbonds +
            w_rmsd * n_rmsd
        )
        r["score"] = f"{composite_val:.4f}"

    # Sort the ENTIRE population globally by the new Master Score ascending
    rows.sort(key=lambda x: float(x["score"]))
    return rows

def process_and_redefine_workspace_scores(output_dir, top_poses_limit=10, has_reference=False):
    """
    Ranks ALL generated buffer poses, slices the absolute best based on the composite score,
    safely renames the physical PDB files sequentially, and writes the clean scores.csv.
    """
    csv_path = Path(output_dir) / "scores.csv"
    if not csv_path.is_file():
        return

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    # 1. Score and rank ALL generated poses globally
    redefined_rows = apply_composite_scoring(rows, has_reference=has_reference)
    
    # 2. Slice the top configurations strictly based on the Composite Equilibrium
    final_top_rows = redefined_rows[:top_poses_limit]
    discarded_rows = redefined_rows[top_poses_limit:]
    prefix = Path(output_dir).stem

    # 3. Safe File Renaming Protocol (Avoids filename collisions during loop)
    # Step A: Move the isolated true top poses to temporary cache files
    for i, row in enumerate(final_top_rows):
        old_path = Path(output_dir) / row["output_file"]
        temp_path = Path(output_dir) / f"temp_safe_{i}.tmp"
        if old_path.is_file():
            shutil.move(str(old_path), str(temp_path))
        row["temp_path"] = temp_path

    # Step B: Delete all the remaining discarded PDB files safely
    for row in discarded_rows:
        discard_path = Path(output_dir) / row["output_file"]
        if discard_path.is_file():
            try:
                discard_path.unlink()
            except Exception:
                pass

    # Step C: Assign final ordered names and restore from temporary cache files
    for i, row in enumerate(final_top_rows):
        new_name = f"{prefix}_ps{i+1:03d}.pdb"
        new_path = Path(output_dir) / new_name
        temp_path = row.pop("temp_path", None)
        
        if temp_path and temp_path.is_file():
            shutil.move(str(temp_path), str(new_path))
            
        # Hard-lock indexing variables for the CSV export
        row["pose_rank"] = str(i + 1)
        row["output_file"] = new_name

    # 4. Write the final selection down to scores.csv
    fieldnames = list(final_top_rows[0].keys())
    if "raw_score" not in fieldnames:
        fieldnames.insert(1, "raw_score")
    if "pose_rank" not in fieldnames:
        fieldnames.insert(0, "pose_rank")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_top_rows)

def pool_worker(args_tuple):
    # Intercept wrapper for the execute_single_docking backend call
    engine_args = args_tuple[:-1]
    final_top_poses = args_tuple[-1]
    
    lf = engine_args[1]
    out_base = engine_args[2]
    ref_atoms = engine_args[9]
    
    # 1. Execute actual docking via the core engine first (Generates Buffer Array)
    status = execute_single_docking(*engine_args)
    
    # 2. Intercept and completely rescore/re-rank/clean ALL generated poses based on composite metric
    ligand_folder = Path(out_base) / Path(lf).stem
    if ligand_folder.is_dir():
        process_and_redefine_workspace_scores(ligand_folder, top_poses_limit=final_top_poses, has_reference=bool(ref_atoms))
        
    return status

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
    print(f" -> Scoring Routine Selected: Master Equal-Weight Composite Evaluation")
    print(f" -> Queue Status: Processing {len(ligand_files)} dynamic input ligand arrays.")
    if ref_atoms:
        print(f" -> Reference Ligand Status: Loaded {len(ref_atoms)} reference atoms cleanly.")
    else:
        print(" -> Reference Ligand Status: None provided or file path unreadable.")
    print("="*60)
    
    # Calculate Deep Buffer Pool: Generates 10x the requested limit (or at least 100) to ensure thorough evaluation
    buffer_poses_pool = 9999999999
    
    tasks = []
    for lf in ligand_files:
        tasks.append((
            receptor_atoms, lf, args.out, tuple(args.center), 
            args.size, args.translation_step, args.rotation_step, 
            args.scoring, buffer_poses_pool, ref_atoms, args.smiles,
            args.top_poses # Final limit parameter appended for the wrapper logic
        ))
        
    with Pool(processes=args.cores) as pool:
        for output_status in pool.imap_unordered(pool_worker, tasks):
            print(f" [✔] Status Update: {output_status}")
            
    print("\n🎉 Parallel docking search and composite ranking filtering completed successfully.")

if __name__ == "__main__":
    main()