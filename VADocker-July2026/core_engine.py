#!/usr/bin/env python3
import math
import heapq
import csv
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, replace
from itertools import product

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

def parse_pdb_line(line: str) -> Atom | None:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")): return None
    try:
        serial = int(line[6:11].strip())
        name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip() if len(line) > 21 else "A"
        res_num = int(line[22:26].strip())
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        element = line[76:78].strip()
        if not element: element = "".join([c for c in name if c.isalpha()])[:2]
        element = element.upper() or "C"
        
        r_h, eps, q = AMBER_PARAMS.get(element, (1.9080, 0.0860, 0.0000))
        if res_name == "FE1" or element == "FE": r_h, eps, q = AMBER_PARAMS["FE1"]
        elif res_name == "HM1":
            r_h, eps, q = AMBER_PARAMS["HM1_N"] if element == "N" else AMBER_PARAMS["HM1_C"]
        elif res_name == "CM1" and element == "S": r_h, eps, q = AMBER_PARAMS["CM1_S"]
        
        return Atom(serial, name, res_name, res_num, chain, element, x, y, z, r_h, eps, q)
    except Exception: return None

def read_pdb(path: Path) -> list[Atom]:
    atoms = []
    if not path.is_file(): return atoms
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            atom = parse_pdb_line(line)
            if atom: atoms.append(atom)
    return atoms

def calculate_heavy_atom_rmsd(pose_atoms: list[Atom], reference_atoms: list[Atom]) -> float:
    pose_heavy = [a for a in pose_atoms if a.element != "H"]
    ref_heavy = [a for a in reference_atoms if a.element != "H"]
    if not pose_heavy or len(pose_heavy) != len(ref_heavy): return 0.0
    sq_dist = sum((p.x - r.x)**2 + (p.y - r.y)**2 + (p.z - r.z)**2 for p, r in zip(pose_heavy, ref_heavy))
    return math.sqrt(sq_dist / len(pose_heavy))

def calculate_empirical_score(ligand_atoms: list[Atom], receptor_atoms: list[Atom]) -> tuple[float, int, int, int]:
    score, clashes, contacts, hbonds = 0.0, 0, 0, 0
    for l_at in ligand_atoms:
        if l_at.element == "H": continue
        for r_at in receptor_atoms:
            if r_at.element == "H": continue
            dist = math.sqrt((l_at.x - r_at.x)**2 + (l_at.y - r_at.y)**2 + (l_at.z - r_at.z)**2)
            if dist > 5.0: continue
            
            combined_rad = VDW_RADIUS.get(l_at.element, 1.7) + VDW_RADIUS.get(r_at.element, 1.7)
            clash_dist = 0.65 * combined_rad
            
            if dist < clash_dist:
                clashes += 1
                score += 30.0 * (clash_dist - dist + 0.1)
            elif dist <= 4.5:
                contacts += 1
                score -= 1.0 / max(dist, 0.1)
                if l_at.element in HYDROPHOBIC and r_at.element in HYDROPHOBIC and 3.2 <= dist <= 4.2:
                    score -= 0.25
                if l_at.element in POLAR and r_at.element in POLAR and 2.4 <= dist <= 3.4:
                    hbonds += 1
                    score -= 1.5
    return score, clashes, contacts, hbonds

def calculate_amber_score(ligand_atoms: list[Atom], receptor_atoms: list[Atom]) -> float:
    total_vdw, total_elec, COULOMB_CONSTANT = 0.0, 0.0, 332.0636
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = math.sqrt((l_at.x - r_at.x)**2 + (l_at.y - r_at.y)**2 + (l_at.z - r_at.z)**2)
            if dist > 10.0: continue
            if dist < 0.8: dist = 0.8
            
            r_min_ij = l_at.r_min_half + r_at.r_min_half
            eps_ij = math.sqrt(l_at.epsilon * r_at.epsilon)
            rm_over_r_6 = (r_min_ij / dist) ** 6
            total_vdw += eps_ij * (rm_over_r_6**2 - 2.0 * rm_over_r_6)
            total_elec += (l_at.charge * r_at.charge * COULOMB_CONSTANT) / (4.0 * dist * dist)
            
    scaled = (total_vdw + total_elec) * 0.20
    return max(min(scaled, 15.0), -15.0)

def execute_single_docking(receptor_atoms: list[Atom], lig_path: Path, out_dir: Path, center: tuple[float, float, float], size: float, step: float, rotation_step: float, scoring_mode: str, top_poses: int, ref_atoms: list[Atom] | None = None) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    lig_atoms = read_pdb(lig_path)
    if not lig_atoms: return f"Skipped {lig_path.name}: Empty array structural maps."
    
    margin = size / 2.0
    active_site = [a for a in receptor_atoms if 
                   center[0]-margin <= a.x <= center[0]+margin and
                   center[1]-margin <= a.y <= center[1]+margin and
                   center[2]-margin <= a.z <= center[2]+margin]
    
    x_range = [center[0] - margin + i*step for i in range(int(size/step)+1)]
    y_range = [center[1] - margin + i*step for i in range(int(size/step)+1)]
    z_range = [center[2] - margin + i*step for i in range(int(size/step)+1)]
    
    lx = [a.x for a in lig_atoms]; ly = [a.y for a in lig_atoms]; lz = [a.z for a in lig_atoms]
    cx, cy, cz = sum(lx)/len(lx), sum(ly)/len(ly), sum(lz)/len(lz)
    
    results = []
    # Clean rotation vector pre-sampling matrix
    angles = [float(i) for i in range(0, 360, int(rotation_step))]
    
    for tx, ty, tz in product(x_range, y_range, z_range):
        for rx in angles:
            rad = math.radians(rx)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            # Optimized on-the-fly matrix mapping rotation track transformation pass
            rotated_moved = []
            for a in lig_atoms:
                dx, dy, dz = a.x - cx, a.y - cy, a.z - cz
                nx = dx * cos_a - dy * sin_a
                ny = dx * sin_a + dy * cos_a
                rotated_moved.append(replace(a, x=nx + tx, y=ny + ty, z=dz + tz))
                
            if scoring_mode == "amber":
                score = calculate_amber_score(rotated_moved, active_site)
                clashes, contacts, hbonds = 0, 0, 0
            else:
                score, clashes, contacts, hbonds = calculate_empirical_score(rotated_moved, active_site)
                
            if scoring_mode == "empirical" and clashes > 10: continue
            
            heapq.heappush(results, (score, clashes, contacts, hbonds, rotated_moved))
            
    csv_out = out_dir / "scores.csv"
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pose_rank", "score", "clashes", "contacts", "hbonds", "rmsd", "output_file"])
        for idx in range(1, top_poses + 1):
            if not results: break
            score, cls, cnt, hb, pose = heapq.heappop(results)
            rmsd_val = calculate_heavy_atom_rmsd(pose, ref_atoms) if ref_atoms else 0.0
            pose_file = out_dir / f"pose_{idx:03d}.pdb"
            
            lines = [f"HETATM{a.serial:>5} {a.name:<4} {a.res_name:<3} {a.chain}{a.res_num:>4}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00           {a.element:<2}" for a in pose]
            pose_file.write_text("\n".join(lines) + "\nEND\n")
            writer.writerow([idx, f"{score:.4f}", cls, cnt, hb, f"{rmsd_val:.4f}", pose_file.name])
            
    return f"Finished processing ligand layout targeting folder: {out_dir.name}"