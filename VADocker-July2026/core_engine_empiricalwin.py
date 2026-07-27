#!/usr/bin/env python3
import math
import heapq
import csv
import sys
import time
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, replace
from itertools import product
from typing import Optional, List, Tuple

MASS_BY_ELEMENT = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.904,
}

VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
}

AMBER_PARAMS = {
    "C":     (1.9080, 0.0860, -0.15), "H":     (1.4870, 0.0157,  0.06),
    "O":     (1.6612, 0.2100, -0.45), "N":     (1.8240, 0.1700, -0.30),
    "S":     (2.0000, 0.2500,  0.05), "P":     (2.1000, 0.2000,  0.20),
    "F":     (1.7500, 0.0610, -0.20), "CL":    (1.9480, 0.2650, -0.15),
    "BR":    (2.2200, 0.3200, -0.10), "I":     (2.3500, 0.4000, -0.05),
    "FE1":   (1.2000, 0.0500,  2.00), "HM1_N": (1.8240, 0.1700, -0.25),
    "HM1_C": (1.9080, 0.0860,  0.05), "CM1_S": (2.0000, 0.2500, -0.60),
}

POLAR = {"N", "O", "S"}
HYDROPHOBIC = {"C", "CL", "BR", "I", "F"}

@dataclass(frozen=True)
class Atom:
    serial: int; name: str; res_name: str; res_num: int; chain: str; element: str
    x: float; y: float; z: float
    r_min_half: float = 1.9080; epsilon: float = 0.0860; charge: float = 0.0000

def parse_pdb_line(line: str) -> Optional[Atom]:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")): return None
    try:
        serial = int(line[6:11].strip())
        name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip() if len(line) > 21 else "A"
        res_num = int(line[22:26].strip())
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        
        element = line[76:78].strip().upper()
        if not element:
            raw_alpha = "".join([c for c in name if c.isalpha()]).upper()
            if raw_alpha.startswith("CL"): element = "CL"
            elif raw_alpha.startswith("BR"): element = "BR"
            elif raw_alpha.startswith("FE"): element = "FE"
            elif raw_alpha: element = raw_alpha[0]
            
        if not element: element = "C"
        
        r_h, eps, q = AMBER_PARAMS.get(element, (1.9080, 0.0860, 0.0000))
        if res_name == "FE1" or element == "FE": r_h, eps, q = AMBER_PARAMS["FE1"]
        elif res_name == "HM1":
            r_h, eps, q = AMBER_PARAMS["HM1_N"] if element == "N" else AMBER_PARAMS["HM1_C"]
        elif res_name == "CM1" and element == "S": r_h, eps, q = AMBER_PARAMS["CM1_S"]
        
        return Atom(serial, name, res_name, res_num, chain, element, x, y, z, r_h, eps, q)
    except Exception: return None

def read_pdb(path: Path) -> List[Atom]:
    atoms = []
    if not path.is_file(): return atoms
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            atom = parse_pdb_line(line)
            if atom: atoms.append(atom)
    return atoms

def parse_atoms_from_pdb_block(block: str) -> List[Atom]:
    atoms = []
    for line in block.splitlines():
        atom = parse_pdb_line(line)
        if atom: atoms.append(atom)
    return atoms

def check_parameter_mismatches(ligand_name: str, smiles_str: Optional[str], parsed_atoms: List[Atom]):
    if not smiles_str:
        return
    try:
        from rdkit import Chem
        tmpl_mol = Chem.MolFromSmiles(smiles_str)
        if not tmpl_mol: return
        tmpl_heavy_count = Chem.Mol(tmpl_mol).GetNumHeavyAtoms()
        parsed_heavy_count = len([a for a in parsed_atoms if a.element != "H"])
        
        if tmpl_heavy_count != parsed_heavy_count:
            print("\n" + "!" * 80)
            print(f"⚠️  POTENTIAL PARAMETER MISMATCH DETECTED FOR LIGAND: {ligand_name}")
            print(f"   -> Enforced SMILES Heavy Atoms: {tmpl_heavy_count}")
            print(f"   -> Coordinates File Heavy Atoms: {parsed_heavy_count}")
            print("-" * 80)
            print("💡 POSSIBLE CAUSE:")
            print("   The shell history might be passing an old '--smiles' template string.")
            print("\n🛠️  SUGGESTED SOLUTION:")
            print("   Update the '--smiles' argument value to match the true layout or omit it.")
            print("!" * 80 + "\n")
            time.sleep(3)
    except Exception: pass

def generate_obabel_conformers(ligand_file: Path, smiles_str: Optional[str] = None) -> List[List[Atom]]:
    if not shutil.which("obabel"):
        raise RuntimeError("Open Babel binary ('obabel') not found in environment PATH.")
        
    in_ext = ligand_file.suffix.lstrip('.').lower()
    temp_out = ligand_file.parent / f"tmpl_ensemble_{ligand_file.stem}.pdb"
    
    cmd = [
        "obabel", f"-i{in_ext}", str(ligand_file), "-opdb", 
        "-O", str(temp_out),
        "--conformer", "--nconf", "50", "--score", "energy"
    ]
    
    try:
        if smiles_str:
            fallback_atoms = read_pdb(ligand_file) if in_ext == "pdb" else []
            if not fallback_atoms and in_ext == "mol2":
                try:
                    txt = ligand_file.read_text(errors="ignore")
                    fallback_atoms = [a for a in parse_atoms_from_pdb_block(txt) if a.element != "H"]
                except Exception: pass
                
            if fallback_atoms:
                check_parameter_mismatches(ligand_file.name, smiles_str, fallback_atoms)
                
            cmd = [
                "obabel", "-ismi", "-opdb", 
                "-O", str(temp_out),
                "--gen3d", "--conformer", "--nconf", "50", "--score", "energy"
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=f"{smiles_str}\n")
        else:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
        if not temp_out.is_file() or temp_out.stat().st_size == 0:
            raise FileNotFoundError("OpenBabel completed but output file is empty/missing.")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n    | ⚠️ OpenBabel conformer generation failed for {ligand_file.name}. Falling back to rigid input structure.")
        try:
            shutil.copy2(ligand_file, temp_out)
        except Exception: 
            pass
        
    conformer_sets = []
    if temp_out.is_file():
        content = temp_out.read_text(errors="ignore")
        models = content.split("ENDMDL")
        for model in models:
            if "ATOM" in model or "HETATM" in model:
                atoms = parse_atoms_from_pdb_block(model)
                if atoms:
                    conformer_sets.append(atoms)
        temp_out.unlink(missing_ok=True)
        
    if not conformer_sets:
        fallback_atoms = read_pdb(ligand_file)
        if fallback_atoms:
            conformer_sets.append(fallback_atoms)
            
    return conformer_sets

def parse_heavy_atom_coordinates_from_list(atom_list: List[Atom]) -> List[Tuple[float, float, float]]:
    return [(a.x, a.y, a.z) for a in atom_list if a.element != "H"]

def calculate_rmsd(coords_a: List[Tuple[float, float, float]], coords_b: List[Tuple[float, float, float]]) -> float:
    if not coords_a or not coords_b or len(coords_a) != len(coords_b): return float('inf')
    sq_dist_sum = sum((xa - xb)**2 + (ya - yb)**2 + (za - zb)**2 for (xa, ya, za), (xb, yb, zb) in zip(coords_a, coords_b))
    return math.sqrt(sq_dist_sum / len(coords_a))

def calculate_heavy_atom_rmsd(pose_atoms: List[Atom], reference_atoms: List[Atom]) -> float:
    pose_heavy = [a for a in pose_atoms if a.element != "H"]
    ref_heavy = [a for a in reference_atoms if a.element != "H"]
    if not pose_heavy or not ref_heavy: return 0.0
    
    sq_dist_sum = 0.0
    matched_ref_indices = set()
    
    for p_at in pose_heavy:
        min_sq_dist = float('inf')
        best_ref_idx = -1
        
        for idx, r_at in enumerate(ref_heavy):
            if idx in matched_ref_indices: continue
            if r_at.element != p_at.element: continue
            
            dx, dy, dz = p_at.x - r_at.x, p_at.y - r_at.y, p_at.z - r_at.z
            d_sq = dx*dx + dy*dy + dz*dz
            if d_sq < min_sq_dist:
                min_sq_dist = d_sq
                best_ref_idx = idx
                
        if best_ref_idx != -1:
            sq_dist_sum += min_sq_dist
            matched_ref_indices.add(best_ref_idx)
        else:
            min_any_dist = float('inf')
            for idx, r_at in enumerate(ref_heavy):
                dx, dy, dz = p_at.x - r_at.x, p_at.y - r_at.y, p_at.z - r_at.z
                d_sq = dx*dx + dy*dy + dz*dz
                if d_sq < min_any_dist: min_any_dist = d_sq
            sq_dist_sum += min_any_dist
            
    return math.sqrt(sq_dist_sum / len(pose_heavy))

def calculate_empirical_score(ligand_atoms: List[Atom], receptor_atoms: List[Atom]) -> Tuple[float, int, int, int]:
    score, clashes, contacts, hbonds = 0.0, 0, 0, 0
    for l_at in ligand_atoms:
        if l_at.element == "H": continue
        l_rad = VDW_RADIUS.get(l_at.element, 1.70)
        
        for r_at in receptor_atoms:
            if r_at.element == "H": continue
            dx, dy, dz = l_at.x - r_at.x, l_at.y - r_at.y, l_at.z - r_at.z
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq > 25.0: continue
            
            r_rad = VDW_RADIUS.get(r_at.element, 1.70)
            combined_rad = l_rad + r_rad
            clash_dist = 0.65 * combined_rad
            
            if dist_sq < clash_dist * clash_dist:
                clashes += 1
                score += 30.0 * (clash_dist - math.sqrt(dist_sq) + 0.1)
            elif dist_sq <= 20.25:
                distance = math.sqrt(dist_sq)
                contacts += 1
                score -= 1.0 / max(distance, 0.1)
                if l_at.element in HYDROPHOBIC and r_at.element in HYDROPHOBIC and 3.2 <= distance <= 4.2:
                    score -= 0.25
                if l_at.element in POLAR and r_at.element in POLAR and 2.4 <= distance <= 3.4:
                    hbonds += 1
                    score -= 1.5
    return score, clashes, contacts, hbonds

def calculate_amber_score(ligand_atoms: List[Atom], receptor_atoms: List[Atom]) -> Tuple[float, int]:
    total_vdw, total_elec, COULOMB_CONSTANT = 0.0, 0.0, 332.0636
    clashes = 0
    for l_at in ligand_atoms:
        l_rad = VDW_RADIUS.get(l_at.element, 1.70)
        for r_at in receptor_atoms:
            dx, dy, dz = l_at.x - r_at.x, l_at.y - r_at.y, l_at.z - r_at.z
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq > 100.0: continue
            
            r_rad = VDW_RADIUS.get(r_at.element, 1.70)
            combined_rad = l_rad + r_rad
            if dist_sq < (0.65 * combined_rad) ** 2:
                if l_at.element != "H" and r_at.element != "H":
                    clashes += 1
            
            dist = math.sqrt(dist_sq)
            if dist < 0.8: dist = 0.8
            
            r_min_ij = l_at.r_min_half + r_at.r_min_half
            eps_ij = math.sqrt(l_at.epsilon * r_at.epsilon)
            rm_over_r_6 = (r_min_ij / dist) ** 6
            total_vdw += eps_ij * (rm_over_r_6**2 - 2.0 * rm_over_r_6)
            total_elec += (l_at.charge * r_at.charge * COULOMB_CONSTANT) / (4.0 * dist_sq)
            
    scaled = (total_vdw + total_elec) * 0.20
    final_score = max(min(scaled, 15.0), -15.0)
    return final_score, clashes

def draw_progress_bar(current: int, total: int, prefix: str = ""):
    bar_length = 30
    fraction = current / max(total, 1)
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    sys.stdout.write(f"\r{prefix} [{arrow}{padding}] {fraction * 100:.1f}% ({current}/{total})")
    sys.stdout.flush()

def execute_single_docking(receptor_atoms: List[Atom], lig_path: Path, out_dir: Path, center: tuple, size: float, step: float, rotation_step: float, scoring_mode: str, top_poses: int, ref_atoms: Optional[List[Atom]] = None, smiles_str: Optional[str] = None) -> str:
    t_start = time.time()
    ligand_base_name = lig_path.stem
    target_workspace = out_dir / ligand_base_name
    target_workspace.mkdir(parents=True, exist_ok=True)
    
    print(f"\n⚡ Processing Molecule Targets: {lig_path.name}")
    
    # --- Step 1: Conformer Generation ---
    t_s1 = time.time()
    print(f" -> [1/5] Generating multi-conformer workspace ensemble using obabel (20 kcal/mol window)...")
    conformer_atom_sets = generate_obabel_conformers(lig_path, smiles_str)
    print(f"    | Generated {len(conformer_atom_sets)} stable structures. Time taken: {time.time() - t_s1:.2f}s")
    
    # --- Step 2: Pocket Geometry Isolation ---
    t_s2 = time.time()
    print(f" -> [2/5] Isolating pocket geometry boundary matrix points...")
    margin = (size / 2.0)
    clash_margin = margin + 5.0
    
    active_site = [a for a in receptor_atoms if 
                   center[0]-clash_margin <= a.x <= center[0]+clash_margin and
                   center[1]-clash_margin <= a.y <= center[1]+clash_margin and
                   center[2]-clash_margin <= a.z <= center[2]+clash_margin]
    
    x_range = [center[0] - margin + i*step for i in range(int(size/step)+1)]
    y_range = [center[1] - margin + i*step for i in range(int(size/step)+1)]
    z_range = [center[2] - margin + i*step for i in range(int(size/step)+1)]
    
    raw_combinations = list(product(x_range, y_range, z_range))
    grid_combinations = []
    
    for gx, gy, gz in raw_combinations:
        too_close = False
        for r_at in active_site:
            if r_at.element == "H": continue
            dx, dy, dz = gx - r_at.x, gy - r_at.y, gz - r_at.z
            if (dx*dx + dy*dy + dz*dz) < 4.0: 
                too_close = True
                break
        if not too_close:
            grid_combinations.append((gx, gy, gz))
            
    input_heavy_coords = parse_heavy_atom_coordinates_from_list(conformer_atom_sets[0])
    print(f"    | Grid filtering completed. Active nodes remaining: {len(grid_combinations)} / {len(raw_combinations)}")
    print(f"    | Selection mapped successfully. Time taken: {time.time() - t_s2:.2f}s")
    
    # --- Step 3: Grid Search Transformation ---
    t_s3 = time.time()
    print(f" -> [3/5] Running docking transformations on multi-conformer grids...")
    
    raw_heap = []
    angles = [float(i) for i in range(0, 360, int(rotation_step))]
    total_tasks = len(conformer_atom_sets) * len(grid_combinations)
    task_idx = 0
    pose_counter = 0  
    
    for lig_atoms in conformer_atom_sets:
        lx = [a.x for a in lig_atoms]; ly = [a.y for a in lig_atoms]; lz = [a.z for a in lig_atoms]
        cx, cy, cz = sum(lx)/len(lx), sum(ly)/len(ly), sum(lz)/len(lz)
        
        for tx, ty, tz in grid_combinations:
            task_idx += 1
            if task_idx % 25 == 0 or task_idx == total_tasks:
                draw_progress_bar(task_idx, total_tasks, prefix="    | Grid Search")
                
            for rx in angles:
                rad = math.radians(rx)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                rotated_moved = []
                for a in lig_atoms:
                    dx, dy, dz = a.x - cx, a.y - cy, a.z - cz
                    nx = dx * cos_a - dy * sin_a
                    ny = dx * sin_a + dy * cos_a
                    rotated_moved.append(replace(a, x=nx + tx, y=ny + ty, z=dz + tz))
                    
                if scoring_mode == "amber":
                    score, clashes = calculate_amber_score(rotated_moved, active_site)
                    contacts, hbonds = 0, 0
                else:
                    score, clashes, contacts, hbonds = calculate_empirical_score(rotated_moved, active_site)
                    
                if clashes > 5: 
                    continue
                
                heapq.heappush(raw_heap, (score, clashes, contacts, hbonds, pose_counter, rotated_moved))
                pose_counter += 1
                
    print(f"\n    | Grid search iterations completed. Time taken: {time.time() - t_s3:.2f}s")
    
    # --- Step 4: Redundancy Filtering ---
    t_s4 = time.time()
    print(f" -> [4/5] Filtering duplicate conformations (1.5 Å RMSD cutoff)...")
    unique_sorted_poses = []
    total_heap = len(raw_heap)
    heap_idx = 0
    duplicate_count = 0
    
    while raw_heap:
        score, cls, cnt, hb, _, current_pose = heapq.heappop(raw_heap)
        heap_idx += 1
        if heap_idx % 50 == 0 or heap_idx == total_heap:
            draw_progress_bar(heap_idx, total_heap, prefix="    | Filter Poses")
            
        current_coords = parse_heavy_atom_coordinates_from_list(current_pose)
        is_duplicate = False
        for _, _, _, _, _, unique_coords in unique_sorted_poses:
            if calculate_rmsd(current_coords, unique_coords) < 1.5:
                is_duplicate = True
                duplicate_count += 1
                break
        if not is_duplicate:
            unique_sorted_poses.append((score, cls, cnt, hb, current_pose, current_coords))
            if len(unique_sorted_poses) >= top_poses: 
                duplicate_count += len(raw_heap)
                break
    print(f"\n    | Filtering operations finished. Time taken: {time.time() - t_s4:.2f}s")

    # --- Step 5: File IO Export ---
    t_s5 = time.time()
    print(f" -> [5/5] Re-indexing workspace target outputs under condensed formats...")
    csv_out = target_workspace / "scores.csv"
    total_exports = len(unique_sorted_poses)
    
    top_scores = []
    top_ref_rmsds = []
    
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pose_rank", "score", "clashes", "contacts", "hbonds", "rmsd_to_input", "rmsd_to_reference", "output_file"])
        
        for rank, (score, cls, cnt, hb, pose, pose_coords) in enumerate(unique_sorted_poses, 1):
            draw_progress_bar(rank, total_exports, prefix="    | Writing Files")
            
            input_rmsd = calculate_rmsd(pose_coords, input_heavy_coords)
            reference_rmsd = calculate_heavy_atom_rmsd(pose, ref_atoms) if ref_atoms else 0.0
            
            top_scores.append(score)
            top_ref_rmsds.append(reference_rmsd)
            
            pose_filename = f"{ligand_base_name}_ps{rank:03d}.pdb"
            pose_file = target_workspace / pose_filename
            
            lines = [f"HETATM{a.serial:>5} {a.name:<4} {a.res_name:<3} {a.chain}{a.res_num:>4}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00           {a.element:<2}" for a in pose]
            pose_file.write_text("\n".join(lines) + "\nEND\n")
            writer.writerow([rank, f"{score:.4f}", cls, cnt, hb, f"{input_rmsd:.4f}", f"{reference_rmsd:.4f}", pose_filename])
            
    print(f"\n    | File export logs completed successfully. Time taken: {time.time() - t_s5:.2f}s")
    
    print("\n" + "="*50)
    print(f"📊 DOCKING RESULTS SUMMARY FOR: {lig_path.name}")
    print("="*50)
    if top_scores:
        print(f"  • Best Docking Score    : {top_scores[0]:.4f} (Lower is better)")
        print(f"  • Worst Saved Score     : {top_scores[-1]:.4f}")
        if ref_atoms:
            print(f"  • Top Pose Ref RMSD     : {top_ref_rmsds[0]:.4f} Å")
            print(f"  • Lowest Ref RMSD Found : {min(top_ref_rmsds):.4f} Å")
        else:
            print(f"  • Top Pose Ref RMSD     : N/A (No reference ligand matched)")
    else:
        print("  • No viable poses survived clash constraints mapping filters.")
    print(f"  • Duplicate Poses Purged: {duplicate_count} orientations")
    print(f"  • Saved Clean Profiles  : {total_exports} pose files")
    print("="*50)
    
    print(f"🎉 Total Execution Time for {lig_path.name}: {time.time() - t_start:.2f}s\n")
    return f"Processed ligand: {lig_path.name} -> Target Folder: {target_workspace.name}"