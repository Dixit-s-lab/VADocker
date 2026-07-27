#!/usr/bin/env python3
import sys
import os
import argparse
import csv
import shutil
import tempfile
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from core_engine_empirical import read_pdb, execute_single_docking

# Pre-flight Check: Ensure GNINA binary is accessible in the environment PATH
GNINA_PATH = shutil.which("gnina")
GNINA_AVAILABLE = GNINA_PATH is not None

def get_gnina_affinity(receptor_path, pose_path):
    """
    Calls the GNINA executable in --score_only mode for a specific protein-ligand complex.
    Parses the stdout to extract the predicted CNN affinity (pKd).
    """
    if not GNINA_AVAILABLE:
        return 0.0
        
    cmd = [
        GNINA_PATH, 
        "--receptor", str(receptor_path), 
        "--ligand", str(pose_path), 
        "--score_only",
        "--quiet"
    ]
    
    try:
        # Run GNINA and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if "CNN affinity" in line:
                # Example output line: "CNN affinity: 6.5432"
                parts = line.split(":")
                if len(parts) > 1:
                    return float(parts[1].strip())
    except Exception as e:
        pass
    
    return 0.0

def min_max_scale(val, val_list):
    min_v, max_v = min(val_list), max(val_list)
    if max_v == min_v:
        return 0.0
    return (val - min_v) / (max_v - min_v)

def apply_composite_scoring(rows, has_reference=False, use_ml=False):
    if not rows:
        return rows

    raw_scores = [float(r.get("raw_score", r.get("score", 0.0))) for r in rows]
    clashes = [float(r.get("clashes", 0.0)) for r in rows]
    contacts = [float(r.get("contacts", 0.0)) for r in rows]
    hbonds = [float(r.get("hbonds", 0.0)) for r in rows]
    rmsds = [float(r.get("rmsd_to_reference", 0.0)) for r in rows]
    ml_scores = [float(r.get("gnina_affinity", 0.0)) for r in rows]

    valid_rmsd = has_reference and any(r > 0.0 for r in rmsds)

    params_count = 4
    if valid_rmsd: params_count += 1
    if use_ml and GNINA_AVAILABLE: params_count += 1
    
    w = 1.0 / params_count

    for i, r in enumerate(rows):
        n_raw = min_max_scale(raw_scores[i], raw_scores)
        n_clashes = min_max_scale(clashes[i], clashes)
        n_contacts = min_max_scale(contacts[i], contacts)
        n_hbonds = 1.0 - min_max_scale(hbonds[i], hbonds) 
        n_rmsd = min_max_scale(rmsds[i], rmsds) if valid_rmsd else 0.0
        
        # GNINA Score (pKd): Higher affinity is better, so we invert it for the "lower is better" composite metric
        n_ml = (1.0 - min_max_scale(ml_scores[i], ml_scores)) if (use_ml and GNINA_AVAILABLE) else 0.0

        r["raw_score"] = f"{raw_scores[i]:.4f}"

        composite_val = (
            w * n_raw +
            w * n_clashes +
            w * n_contacts +
            w * n_hbonds +
            (w * n_rmsd if valid_rmsd else 0.0) +
            (w * n_ml if (use_ml and GNINA_AVAILABLE) else 0.0)
        )
        r["score"] = f"{composite_val:.4f}"

    rows.sort(key=lambda x: float(x["score"]))
    return rows

def process_and_redefine_workspace_scores(output_dir, receptor_path, top_poses_limit=10, has_reference=False, use_ml=False):
    csv_files = list(Path(output_dir).glob("*.csv"))
    if not csv_files:
        print(f"⚠️ Warning: No CSV file found in temp directory {output_dir}. Rescoring aborted.", flush=True)
        return
    csv_path = csv_files[0]

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    if use_ml:
        if GNINA_AVAILABLE:
            print(f"\n -> Starting Deep Learning Evaluation (GNINA CNN) on {len(rows)} buffer poses...", flush=True)
            total_poses = len(rows)
            for i, r in enumerate(rows):
                pose_path = Path(output_dir) / r["output_file"]
                if pose_path.is_file():
                    affinity = get_gnina_affinity(receptor_path, pose_path)
                    r["gnina_affinity"] = f"{affinity:.3f}"
                else:
                    r["gnina_affinity"] = "0.0"
                
                pct = (i + 1) / total_poses * 100
                bar_len = int(pct / 100 * 30)
                bar = "=" * bar_len + ">" + " " * (30 - bar_len)
                print(f"    | GNINA Rescoring [{bar}] {pct:.1f}% ({i+1}/{total_poses})", flush=True)
        else:
            print("\n⚠️ Warning: ML requested but 'gnina' binary is not found in PATH. Skipping ML evaluation.", flush=True)
            for r in rows: r["gnina_affinity"] = "0.0"
            use_ml = False

    redefined_rows = apply_composite_scoring(rows, has_reference=has_reference, use_ml=use_ml)
    
    final_top_rows = redefined_rows[:top_poses_limit]
    discarded_rows = redefined_rows[top_poses_limit:]
    prefix = Path(output_dir).stem

    for i, row in enumerate(final_top_rows):
        old_path = Path(output_dir) / row["output_file"]
        temp_path = Path(output_dir) / f"temp_safe_{i}.tmp"
        if old_path.is_file():
            shutil.move(str(old_path), str(temp_path))
        row["temp_path"] = temp_path

    for row in discarded_rows:
        discard_path = Path(output_dir) / row["output_file"]
        if discard_path.is_file():
            try: discard_path.unlink()
            except Exception: pass

    for i, row in enumerate(final_top_rows):
        new_name = f"{prefix}_ps{i+1:03d}.pdb"
        new_path = Path(output_dir) / new_name
        temp_path = row.pop("temp_path", None)
        if temp_path and temp_path.is_file():
            shutil.move(str(temp_path), str(new_path))
        row["pose_rank"] = str(i + 1)
        row["output_file"] = new_name

    fieldnames = list(final_top_rows[0].keys())
    if "raw_score" not in fieldnames: fieldnames.insert(1, "raw_score")
    if "pose_rank" not in fieldnames: fieldnames.insert(0, "pose_rank")

    export_csv = Path(output_dir) / "scores.csv"
    with export_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_top_rows)
        
    if csv_path.name != "scores.csv":
        try: csv_path.unlink()
        except Exception: pass

def pool_worker(args_tuple):
    engine_args = list(args_tuple[:11])
    final_top_poses = args_tuple[11]
    use_ml = args_tuple[12]
    receptor_file = args_tuple[13]
    
    lf = engine_args[1]
    out_base = Path(engine_args[2])
    ref_atoms = engine_args[9]
    
    ligand_name = Path(lf).stem
    final_dest = out_base / ligand_name
    
    with tempfile.TemporaryDirectory() as local_tmp:
        engine_args[2] = Path(local_tmp)
        
        status = execute_single_docking(*engine_args)
        
        print(f"\n -> Pre-docking complete. Analyzing local temp buffer: {local_tmp}", flush=True)
        
        tmp_ligand_folder = Path(local_tmp) / ligand_name
        if tmp_ligand_folder.is_dir():
            process_and_redefine_workspace_scores(
                tmp_ligand_folder, 
                receptor_path=receptor_file, 
                top_poses_limit=final_top_poses, 
                has_reference=bool(ref_atoms),
                use_ml=use_ml
            )
            
            print(f"\n -> Exporting top finalized poses to Google Drive: {final_dest}", flush=True)
            final_dest.mkdir(parents=True, exist_ok=True)
            for item in tmp_ligand_folder.iterdir():
                shutil.copy2(item, final_dest / item.name)
        else:
            print(f"⚠️ Error: Core engine failed to create workspace folder inside {local_tmp}", flush=True)
                
    return status

def main():
    parser = argparse.ArgumentParser(description="VADocker Parallel Engine - Deep Learning GNINA Pre-Ranking")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ligands", type=Path, help="Directory folder containing all ligand structure files.")
    group.add_argument("--ligand", type=Path, help="Explicit singular ligand target configuration file.")
    
    parser.add_argument("--receptor", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center", nargs=3, type=float, required=True)
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--translation-step", type=float, default=1.5)
    parser.add_argument("--rotation-step", type=float, default=20.0)
    parser.add_argument("--top-poses", type=int, default=10)
    parser.add_argument("--scoring", choices=["empirical", "amber", "gnina_ml"], default="empirical")
    parser.add_argument("--cores", type=int, default=cpu_count())
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--smiles", type=str, default=None)

    args = parser.parse_args()
    
    if not args.receptor.is_file():
        print(f"Error: Receptor configuration layout target missing at {args.receptor}", flush=True)
        sys.exit(1)
        
    receptor_atoms = read_pdb(args.receptor)
    ref_atoms = read_pdb(args.reference) if args.reference and args.reference.is_file() else None
    
    if args.ligand:
        ligand_files = [args.ligand]
    else:
        ligand_files = sorted(f for f in args.ligands.iterdir() if f.is_file() and f.suffix.lower() in {".pdb", ".mol2", ".sdf"})

    print("="*60, flush=True)
    print(f"🧬 LATTICE SEARCH RUN STARTED | Engine Track: Parallel CPU Mode ({args.cores} cores)", flush=True)
    print(f" -> Master Score Mode: {args.scoring.upper()} Composite", flush=True)
    print(f" -> ML Model Status: {'Active (GNINA CNN)' if (args.scoring == 'gnina_ml' and GNINA_AVAILABLE) else 'Inactive'}", flush=True)
    print("="*60, flush=True)
    
    buffer_poses_pool = 9999999
    use_ml = (args.scoring == "gnina_ml")
    
    tasks = []
    for lf in ligand_files:
        tasks.append((
            receptor_atoms, lf, args.out, tuple(args.center), 
            args.size, args.translation_step, args.rotation_step, 
            "empirical", buffer_poses_pool, ref_atoms, args.smiles,
            args.top_poses, use_ml, args.receptor
        ))
        
    with Pool(processes=args.cores) as pool:
        for output_status in pool.imap_unordered(pool_worker, tasks):
            pass
            
    print("\n🎉 Parallel docking search and ML composite ranking completed successfully.", flush=True)

if __name__ == "__main__":
    main()