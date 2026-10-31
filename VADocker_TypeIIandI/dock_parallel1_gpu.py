#!/usr/bin/env python3
import sys
import os
import re
import argparse
import csv
import shutil
import math
import subprocess
import time
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Unified Core Engine for both Dry and Hydrated Docking
from core_engine_empirical_gpu import Atom, read_pdb, execute_single_docking


# =============================================================================
# REAL-TIME DUAL LOGGING STREAM (SLURM + PER-LIGAND FILE)
# =============================================================================

class TeeLogger:
    """
    Duplicates stdout to both terminal/Slurm stdout and a per-ligand log file.
    Enforces immediate flushing so Slurm's .out updates without buffering delays.
    """
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_file = None

    def __enter__(self):
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()


# =============================================================================
# LIGAND PROPERTY UTILITIES (MOLECULAR WEIGHT SORTING)
# =============================================================================

def get_ligand_molecular_weight(lig_path: Path) -> float:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        ext = lig_path.suffix.lower()
        mol = None

        if ext == ".mol2":
            mol = Chem.MolFromMol2File(str(lig_path), sanitize=False, removeHs=False)
            if mol is None and shutil.which("obabel"):
                res = subprocess.run(
                    ["obabel", "-imol2", str(lig_path), "-osdf"],
                    capture_output=True, text=True
                )
                if res.returncode == 0 and res.stdout:
                    mol = Chem.MolFromMolBlock(res.stdout, sanitize=False, removeHs=False)
        elif ext in (".sdf", ".sd"):
            suppl = Chem.SDMolSupplier(str(lig_path), sanitize=False, removeHs=False)
            if suppl and len(suppl) > 0:
                mol = suppl[0]
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(str(lig_path), sanitize=False, removeHs=False)

        if mol is not None:
            mol.UpdatePropertyCache(strict=False)
            return float(Descriptors.MolWt(mol))
    except Exception:
        pass
    return float(lig_path.stat().st_size)


# =============================================================================
# REFERENCE STRUCTURE DISCOVERY & HEAVY-ATOM RMSD UTILITIES
# =============================================================================

def extract_primary_code(name: str) -> str:
    m = re.match(r"^([a-zA-Z0-9]{4})", name)
    if m:
        return m.group(1).upper()
    tokens = re.split(r"[_\-\.]", name)
    return tokens[0].upper() if tokens else name.upper()


def get_all_reference_files(ref_dir: Path) -> List[Path]:
    if not ref_dir or not ref_dir.is_dir():
        return []
    valid_exts = {".pdb", ".mol2", ".sdf", ".sd", ".ent"}
    files = []
    for f in sorted(ref_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_exts:
            stem_low = f.stem.lower()
            if any(k in stem_low for k in ("receptor", "protein", "_rec", "apo")):
                continue
            files.append(f)
    return files


def find_reference_file_advanced(
    ref_dir: Path, 
    ligand_name: str, 
    all_files: Optional[List[Path]] = None
) -> Optional[Path]:
    if not ref_dir or not ref_dir.is_dir():
        return None

    if all_files is None:
        all_files = get_all_reference_files(ref_dir)
    if not all_files:
        return None

    lig_code = extract_primary_code(ligand_name)

    # 1. Primary Criterion: matches '<lig_code>_lig_aln.*' or '<ligand_name>_lig_aln.*'
    scheme_targets = [
        f"{lig_code.lower()}_lig_aln",
        f"{lig_code.upper()}_lig_aln",
        f"{ligand_name.lower()}_lig_aln",
        f"{ligand_name.upper()}_lig_aln",
    ]
    for target in scheme_targets:
        for f in all_files:
            if f.stem.lower() == target.lower():
                return f

    if lig_code and len(lig_code) >= 3:
        for f in all_files:
            s_low = f.stem.lower()
            if lig_code.lower() in s_low and "lig_aln" in s_low:
                return f

    # 2. 4-Letter Code Relationships
    if lig_code and len(lig_code) >= 3:
        for f in all_files:
            if f.stem.lower() == lig_code.lower():
                return f

        for f in all_files:
            if extract_primary_code(f.stem) == lig_code:
                return f

        for f in all_files:
            if lig_code.lower() in f.stem.lower():
                return f

    # 3. Full / Partial ligand name matching
    for f in all_files:
        if f.stem.lower() == ligand_name.lower():
            return f

    for f in all_files:
        f_clean = f.stem.lower().replace("_lig", "").replace("_aln", "").replace("_ref", "")
        if f_clean and (f_clean in ligand_name.lower() or ligand_name.lower() in f.stem.lower()):
            return f

    # 4. Fallback if directory contains exactly 1 reference file
    if len(all_files) == 1:
        return all_files[0]

    return None


def load_reference_atoms(ref_path: Path) -> List[Atom]:
    if not ref_path or not ref_path.is_file():
        return []
    ext = ref_path.suffix.lower()
    if ext == ".pdb":
        return read_pdb(ref_path)

    try:
        from rdkit import Chem
        mol = None
        if ext == ".mol2":
            mol = Chem.MolFromMol2File(str(ref_path), sanitize=False, removeHs=False)
            if mol is None and shutil.which("obabel"):
                res = subprocess.run(["obabel", "-imol2", str(ref_path), "-osdf"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout:
                    mol = Chem.MolFromMolBlock(res.stdout, sanitize=False, removeHs=False)
        elif ext in (".sdf", ".sd"):
            suppl = Chem.SDMolSupplier(str(ref_path), sanitize=False, removeHs=False)
            if suppl and len(suppl) > 0:
                mol = suppl[0]

        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            atoms = []
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                elem = atom.GetSymbol().upper()
                name = f"{elem}{i+1}"
                atoms.append(Atom(
                    serial=i+1, name=name, res_name="LIG", res_num=1, chain="A",
                    element=elem, x=float(pos.x), y=float(pos.y), z=float(pos.z)
                ))
            return atoms
    except Exception:
        pass

    return read_pdb(ref_path)


def resolve_ligand_reference(
    ref_dir: Optional[Path],
    common_ref_file: Optional[Path],
    ligand_name: str,
    all_ref_files: List[Path]
) -> Tuple[Optional[Path], Optional[List[Atom]]]:
    matched_file = None

    if common_ref_file and common_ref_file.is_file():
        matched_file = common_ref_file
    elif ref_dir and all_ref_files:
        matched_file = find_reference_file_advanced(ref_dir, ligand_name, all_ref_files)

    if matched_file and matched_file.is_file():
        ref_atoms = load_reference_atoms(matched_file)
        if ref_atoms:
            return matched_file, ref_atoms

    return None, None


def read_heavy_atoms_from_pdb_file(pdb_path: Path, strip_waters: bool = True) -> List[Tuple[str, float, float, float]]:
    atoms = []
    if not pdb_path.is_file():
        return atoms
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    res_name = line[17:20].strip().upper()
                    if strip_waters and res_name in {"HOH", "WAT", "TIP3", "H2O"}:
                        continue
                    name = line[12:16].strip()
                    elem = line[76:78].strip().upper()
                    if not elem:
                        alpha = "".join([c for c in name if c.isalpha()]).upper()
                        if alpha.startswith("CL"): elem = "CL"
                        elif alpha.startswith("BR"): elem = "BR"
                        elif alpha.startswith("FE"): elem = "FE"
                        elif alpha: elem = alpha[0]
                    if not elem: elem = "C"
                    if elem == "H":
                        continue
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((elem, x, y, z))
                except ValueError:
                    continue
    return atoms


def normalize_heavy_atoms(source, strip_waters: bool = True) -> List[Tuple[str, float, float, float]]:
    if not source:
        return []
    if isinstance(source, (str, Path)):
        return read_heavy_atoms_from_pdb_file(Path(source), strip_waters=strip_waters)
    result = []
    for a in source:
        if isinstance(a, tuple):
            elem, x, y, z = a
            res_name = ""
        else:
            elem = getattr(a, "element", "C").upper()
            res_name = getattr(a, "res_name", "").upper()
            x = float(getattr(a, "x", 0.0))
            y = float(getattr(a, "y", 0.0))
            z = float(getattr(a, "z", 0.0))
        if strip_waters and res_name in {"HOH", "WAT", "TIP3", "H2O"}:
            continue
        if elem == "H":
            continue
        result.append((elem, x, y, z))
    return result


def compute_heavy_atom_rmsd(pose_atoms_source, ref_atoms_source, strip_waters: bool = True) -> float:
    pose_heavy = normalize_heavy_atoms(pose_atoms_source, strip_waters=strip_waters)
    ref_heavy = normalize_heavy_atoms(ref_atoms_source, strip_waters=strip_waters)
    if not pose_heavy or not ref_heavy:
        return 0.0

    sq_dist_sum = 0.0
    matched_ref_indices = set()
    for p_elem, px, py, pz in pose_heavy:
        min_sq_dist = float("inf")
        best_ref_idx = -1
        for idx, (r_elem, rx, ry, rz) in enumerate(ref_heavy):
            if idx in matched_ref_indices or r_elem != p_elem:
                continue
            d_sq = (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2
            if d_sq < min_sq_dist:
                min_sq_dist = d_sq
                best_ref_idx = idx
        if best_ref_idx != -1:
            sq_dist_sum += min_sq_dist
            matched_ref_indices.add(best_ref_idx)
        else:
            sq_dist_sum += 16.0
    return math.sqrt(sq_dist_sum / len(pose_heavy))


# =============================================================================
# HEME PLANE & HETEROCYCLE GEOMETRY CLASSIFIER
# =============================================================================

def get_heme_geometry(receptor_atoms: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    fe_pos = None
    cys_sg_pos = None
    pyrrole_coords = []

    for a in receptor_atoms:
        r_name = getattr(a, "res_name", "").upper()
        a_name = getattr(a, "name", "").upper()
        elem = getattr(a, "element", "").upper()

        if r_name in ("HEM", "HEME", "FE", "FE1"):
            if "FE" in a_name or elem == "FE":
                fe_pos = np.array([float(a.x), float(a.y), float(a.z)])
            elif a_name in ("NA", "NB", "NC", "ND") or (elem == "N" and "N" in a_name):
                pyrrole_coords.append([float(a.x), float(a.y), float(a.z)])
        elif "CYS" in r_name and ("SG" in a_name or elem == "S"):
            cys_sg_pos = np.array([float(a.x), float(a.y), float(a.z)])

    if len(pyrrole_coords) >= 4:
        pts = np.array(pyrrole_coords[:4])
        v1 = pts[2] - pts[0]
        v2 = pts[3] - pts[1]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-4:
            return fe_pos, (normal / norm)

    if fe_pos is not None and cys_sg_pos is not None:
        v_axial = fe_pos - cys_sg_pos
        norm = np.linalg.norm(v_axial)
        if norm > 1e-4:
            return fe_pos, (v_axial / norm)

    if fe_pos is not None:
        return fe_pos, np.array([0.0, 0.0, 1.0])

    return None, None


def get_heterocyclic_n_info(ligand_file: Path) -> Tuple[Set[int], Set[int]]:
    try:
        from rdkit import Chem
    except ImportError:
        return set(), set()

    ext = ligand_file.suffix.lower()
    mol = None

    if ext == ".mol2":
        try: mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False, sanitize=True)
        except Exception: mol = None
        if mol is None and shutil.which("obabel"):
            try:
                res = subprocess.run(["obabel", "-imol2", str(ligand_file), "-osdf", "-h"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout:
                    mol = Chem.MolFromMolBlock(res.stdout, removeHs=False)
            except Exception: mol = None
    elif ext in (".sdf", ".sd"):
        try:
            suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
            if suppl and len(suppl) > 0: mol = suppl[0]
        except Exception: mol = None
    elif ext == ".pdb":
        try: mol = Chem.MolFromPDBFile(str(ligand_file), removeHs=False)
        except Exception: mol = None

    if mol is None:
        return set(), set()

    try:
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
    except Exception: pass

    smarts = "[$([n;X2]),$([#7;R;X2;!$([#7+]);!$([#7H])])]"
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return set(), set()

    matches = mol.GetSubstructMatches(query)
    candidate_indices = {m[0] for m in matches}

    unhindered_n = set()
    hindered_n = set()

    for n_idx in candidate_indices:
        atom = mol.GetAtomWithIdx(n_idx)
        ring_nbrs = [nbr for nbr in atom.GetNeighbors() if nbr.IsInRing()]
        is_hindered = False
        for r_nbr in ring_nbrs:
            heavy_nbrs = [nbr2 for nbr2 in r_nbr.GetNeighbors() if nbr2.GetAtomicNum() > 1]
            if len(heavy_nbrs) > 2:
                is_hindered = True
                break
        if is_hindered:
            hindered_n.add(n_idx)
        else:
            unhindered_n.add(n_idx)

    return unhindered_n, hindered_n


def classify_pose_interaction(
    pose_pdb_path: Path,
    fe_pos: Optional[np.ndarray],
    heme_normal: Optional[np.ndarray],
    unhindered_n: Set[int],
    hindered_n: Set[int]
) -> Dict[str, object]:
    coords = []
    with pose_pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                res_name = line[17:20].strip().upper()
                if res_name in {"HOH", "WAT", "TIP3", "H2O"}:
                    continue
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue

    all_het_n = unhindered_n | hindered_n
    if not coords or fe_pos is None or not all_het_n:
        return {
            "fe_n_distance": "N/A",
            "ring_heme_angle": "N/A",
            "interaction_type": "Non-heterocyclic"
        }

    coords_arr = np.array(coords)
    min_dist = float("inf")
    best_n_idx = None
    for n_idx in all_het_n:
        if n_idx < len(coords_arr):
            d = float(np.linalg.norm(coords_arr[n_idx] - fe_pos))
            if d < min_dist:
                min_dist = d
                best_n_idx = n_idx

    if best_n_idx is None:
        return {
            "fe_n_distance": "N/A",
            "ring_heme_angle": "N/A",
            "interaction_type": "Non-heterocyclic"
        }

    dists_from_n = np.linalg.norm(coords_arr - coords_arr[best_n_idx], axis=1)
    ring_atoms = coords_arr[dists_from_n < 2.5]

    if len(ring_atoms) >= 3 and heme_normal is not None:
        centered = ring_atoms - ring_atoms.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        ring_normal = vh[2]
        ring_normal /= np.linalg.norm(ring_normal)
        dot = np.clip(abs(float(np.dot(ring_normal, heme_normal))), 0.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(dot)))
    else:
        angle_deg = 90.0

    if min_dist <= 2.6 and angle_deg >= 45.0:
        if best_n_idx in unhindered_n:
            itype = "Fe-N Coordinated (Unhindered)"
        else:
            itype = "Fe-N Coordinated (Hindered)"
    elif angle_deg <= 35.0 and min_dist <= 4.5:
        itype = "Pi-Pi Stacked"
    else:
        itype = "Distal / Non-Coordinated"

    return {
        "fe_n_distance": round(min_dist, 3),
        "ring_heme_angle": round(angle_deg, 1),
        "interaction_type": itype
    }


# =============================================================================
# HARD CLASH FILTER & EQUAL-WEIGHTAGE COMPOSITE SCORING
# =============================================================================

def get_interaction_tier(interaction_type: str) -> int:
    itype = str(interaction_type).strip()
    if itype == "Fe-N Coordinated (Unhindered)":
        return 0
    elif itype == "Pi-Pi Stacked":
        return 2
    return 1


def filter_and_rank_poses(rows: List[dict], max_clashes: int = 2, has_unhindered_n: bool = False) -> List[dict]:
    if not rows:
        return rows

    parsed_rows = []
    for r in rows:
        row_copy = dict(r)
        raw_val = float(row_copy.get("raw_score", row_copy.get("score", 0.0)))
        clash_val = int(float(row_copy.get("clashes", 0)))
        itype = row_copy.get("interaction_type", "Distal / Non-Coordinated")
        
        row_copy["_raw_numeric"] = raw_val
        row_copy["_clashes_numeric"] = clash_val
        row_copy["_tier"] = get_interaction_tier(itype)

        try:
            row_copy["_dist_numeric"] = float(row_copy.get("fe_n_distance", "nan"))
        except (ValueError, TypeError):
            row_copy["_dist_numeric"] = float("nan")

        try:
            row_copy["_tilt_numeric"] = float(row_copy.get("ring_heme_angle", "nan"))
        except (ValueError, TypeError):
            row_copy["_tilt_numeric"] = float("nan")

        parsed_rows.append(row_copy)

    # 1. Hard steric clash filtering
    valid_poses = [r for r in parsed_rows if r["_clashes_numeric"] <= max_clashes]

    if not valid_poses:
        min_clash = min(r["_clashes_numeric"] for r in parsed_rows)
        valid_poses = [r for r in parsed_rows if r["_clashes_numeric"] <= min_clash + 1]

    # 2. Composite Score vs. Raw Score Calculation Across All Valid Poses
    if has_unhindered_n and valid_poses:
        raw_vals = [r["_raw_numeric"] for r in valid_poses]
        dist_vals = [r["_dist_numeric"] for r in valid_poses if not math.isnan(r["_dist_numeric"])]
        tilt_vals = [r["_tilt_numeric"] for r in valid_poses if not math.isnan(r["_tilt_numeric"])]

        min_raw, max_raw = min(raw_vals), max(raw_vals)
        min_dist = min(dist_vals) if dist_vals else 2.20
        max_dist = max(dist_vals) if dist_vals else 15.0

        tilt_devs = [abs(t - 90.0) for t in tilt_vals] if tilt_vals else [0.0]
        min_dev = min(tilt_devs)
        max_dev = max(tilt_devs)

        for r in valid_poses:
            # 1. Scaled raw score (-ve favored: more negative -> closer to 0.0)
            if max_raw != min_raw:
                s_raw = (r["_raw_numeric"] - min_raw) / (max_raw - min_raw)
            else:
                s_raw = 0.0

            # 2. Scaled distance (smaller distance favored: smaller -> closer to 0.0)
            d_val = r["_dist_numeric"]
            if math.isnan(d_val):
                s_dist = 1.0
            elif max_dist != min_dist:
                s_dist = (d_val - min_dist) / (max_dist - min_dist)
            else:
                s_dist = 0.0

            # 3. Scaled tilt (closer to 90 deg favored: smaller abs(tilt - 90) -> closer to 0.0)
            t_val = r["_tilt_numeric"]
            if math.isnan(t_val):
                s_tilt = 1.0
            else:
                dev = abs(t_val - 90.0)
                if max_dev != min_dev:
                    s_tilt = (dev - min_dev) / (max_dev - min_dev)
                else:
                    s_tilt = 0.0

            comp_score = (s_raw + s_dist + s_tilt) / 3.0
            r["_final_score_numeric"] = comp_score
            r["_is_composite"] = True
    else:
        for r in valid_poses:
            r["_final_score_numeric"] = r["_raw_numeric"]
            r["_is_composite"] = False

    # 3. Sorting hierarchy: Clashes -> Final Score (ascending: lower/more negative is better)
    valid_poses.sort(key=lambda x: (x["_clashes_numeric"], x["_final_score_numeric"]))

    for r in valid_poses:
        raw_f = r.pop("_raw_numeric")
        score_f = r.pop("_final_score_numeric")
        is_comp = r.pop("_is_composite", False)
        r.pop("_clashes_numeric", None)
        r.pop("_tier", None)
        r.pop("_dist_numeric", None)
        r.pop("_tilt_numeric", None)
        r["raw_score"] = f"{raw_f:.4f}"
        if is_comp:
            r["score"] = f"{score_f:.4f}"
        else:
            r["score"] = f"{raw_f:.4f}"

    return valid_poses


def process_and_redefine_workspace_scores(
    output_dir: Path,
    receptor_atoms: list,
    ligand_file: Path,
    top_poses_limit: int = 10,
    max_clashes: int = 2,
    ref_atoms: Optional[list] = None,
    ref_name: Optional[str] = None,
    is_hydrated: bool = False
):
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

    fe_pos, heme_normal = get_heme_geometry(receptor_atoms)
    unhindered_n, hindered_n = get_heterocyclic_n_info(ligand_file)

    # Enable composite coordination scoring ONLY if BOTH active-site Fe AND unhindered N exist
    has_unhindered_coord = (fe_pos is not None) and (len(unhindered_n) > 0)

    for r in rows:
        pose_file = Path(output_dir) / r["output_file"]
        if pose_file.is_file():
            metrics = classify_pose_interaction(pose_file, fe_pos, heme_normal, unhindered_n, hindered_n)
            r["fe_n_distance"] = str(metrics["fe_n_distance"])
            r["ring_heme_angle"] = str(metrics["ring_heme_angle"])
            r["interaction_type"] = str(metrics["interaction_type"])
        else:
            r["fe_n_distance"] = "N/A"
            r["ring_heme_angle"] = "N/A"
            r["interaction_type"] = "Unclassified"

    final_top_rows = filter_and_rank_poses(
        rows, max_clashes=max_clashes, has_unhindered_n=has_unhindered_coord
    )[:top_poses_limit]

    kept_files = {r["output_file"] for r in final_top_rows}
    all_files = {r["output_file"] for r in rows}

    for fname in (all_files - kept_files):
        discard_path = Path(output_dir) / fname
        if discard_path.is_file():
            try: discard_path.unlink()
            except Exception: pass

    prefix = Path(output_dir).stem
    for i, row in enumerate(final_top_rows):
        old_path = Path(output_dir) / row["output_file"]
        temp_path = Path(output_dir) / f"temp_safe_{i}.tmp"
        if old_path.is_file():
            shutil.move(str(old_path), str(temp_path))
        row["temp_path"] = temp_path

    for i, row in enumerate(final_top_rows):
        new_name = f"{prefix}_ps{i+1:03d}.pdb"
        new_path = Path(output_dir) / new_name
        temp_path = row.pop("temp_path", None)
        if temp_path and temp_path.is_file():
            shutil.move(str(temp_path), str(new_path))
        row["pose_rank"] = str(i + 1)
        row["output_file"] = new_name

    # Dynamic RMSD Calculation (stripping waters from docked pose when hydrated)
    if ref_atoms:
        rmsd_basis_desc = f"Reference structure ({ref_name})"
        for row in final_top_rows:
            pose_p = Path(output_dir) / row["output_file"]
            pose_heavy = read_heavy_atoms_from_pdb_file(pose_p, strip_waters=True)
            val = compute_heavy_atom_rmsd(pose_heavy, ref_atoms, strip_waters=True)
            row["rmsd_to_reference"] = f"{val:.4f}"
    else:
        rmsd_basis_desc = f"Top Pose ({prefix}_ps001.pdb)"
        top_pose_p = Path(output_dir) / final_top_rows[0]["output_file"]
        top_pose_heavy = read_heavy_atoms_from_pdb_file(top_pose_p, strip_waters=True)
        for i, row in enumerate(final_top_rows):
            if i == 0:
                row["rmsd_to_reference"] = "0.0000"
            else:
                pose_p = Path(output_dir) / row["output_file"]
                pose_heavy = read_heavy_atoms_from_pdb_file(pose_p, strip_waters=True)
                val = compute_heavy_atom_rmsd(pose_heavy, top_pose_heavy, strip_waters=True)
                row["rmsd_to_reference"] = f"{val:.4f}"

    standard_cols = [
        "pose_rank", "score", "raw_score", "clashes", "contacts", "hbonds",
        "fe_n_distance", "ring_heme_angle", "interaction_type",
        "rmsd_to_reference", "output_file"
    ]
    fieldnames = [c for c in standard_cols if any(c in r for r in final_top_rows)]
    for k in final_top_rows[0].keys():
        if k not in fieldnames and k != "temp_path":
            fieldnames.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_top_rows)

    # Print summary table directly to console and per-ligand log
    print("\n" + "=" * 98)
    print(f"📋 CURATED POSES SUMMARY: {prefix}")
    print(f"   RMSD Basis: {rmsd_basis_desc}")
    if has_unhindered_coord:
        print(f"   Ranking Scheme: Composite Score = 1/3*scaled_raw + 1/3*scaled_dist + 1/3*scaled_tilt")
        print(f"                   (-ve raw scores, smaller distances, and tilts closer to 90° favored)")
    elif fe_pos is None:
        print(f"   Ranking Scheme: Raw Score Only (Non-Heme receptor system)")
    else:
        print(f"   Ranking Scheme: Raw Score Only (No unhindered heterocyclic N)")
    print("=" * 98)
    print(f"{'RANK':<5} {'SCORE':<11} {'RAW SCORE':<11} {'CLASHES':<8} {'DIST(Å)':<10} {'TILT(°)':<10} {'RMSD(Å)':<10} {'INTERACTION TYPE'}")
    print("-" * 98)
    for r in final_top_rows:
        dist_str = f"{float(r['fe_n_distance']):.3f}" if r['fe_n_distance'] != "N/A" else "N/A"
        tilt_str = f"{float(r['ring_heme_angle']):.1f}" if r['ring_heme_angle'] != "N/A" else "N/A"
        rmsd_str = f"{float(r.get('rmsd_to_reference', 0.0)):.4f}" if r.get('rmsd_to_reference') not in (None, "N/A", "") else "N/A"
        print(f"{r['pose_rank']:<5} {r['score']:<11} {r['raw_score']:<11} {r['clashes']:<8} {dist_str:<10} {tilt_str:<10} {rmsd_str:<10} {r['interaction_type']}")
    print("=" * 98 + "\n")


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VADocker GPU Batch Driver (PyTorch Matrix Decomposed)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ligands", type=Path, help="Directory containing multiple ligand files")
    group.add_argument("--ligand", type=Path, help="Single ligand file to dock")
    
    parser.add_argument("--receptor", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center", nargs=3, type=float, required=True)
    parser.add_argument("--size", type=float, default=15.0)
    parser.add_argument("--translation-step", type=float, default=1.5)
    parser.add_argument("--rotation-step", type=float, default=20.0)
    parser.add_argument("--top-poses", type=int, default=10)
    parser.add_argument("--max-clashes", type=int, default=2, help="Max steric clashes permitted before filtering")
    parser.add_argument("--scoring", choices=["empirical", "amber"], default="empirical")
    
    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument("--reference", type=Path, default=None, help="Single reference ligand file (common for all ligands)")
    ref_group.add_argument("--references", type=Path, default=None, help="Directory containing reference ligand structures")
    
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--max-conformers", type=int, default=25)
    parser.add_argument("--energy-window", type=float, default=25.0)
    parser.add_argument("--hydrated", action="store_true", help="Enable field-directed water seeding.")
    parser.add_argument("--e-bulk", type=float, default=-9.5)
    parser.add_argument("--max-waters", type=int, default=10)

    args = parser.parse_args()
    
    receptor_atoms = read_pdb(args.receptor)
    fe_pos_rec, _ = get_heme_geometry(receptor_atoms)
    is_heme_system = fe_pos_rec is not None
    
    ref_dir = None
    common_ref_file = None
    all_ref_files = []

    target_ref = args.references if args.references else args.reference
    if target_ref:
        if target_ref.is_dir():
            ref_dir = target_ref
            all_ref_files = get_all_reference_files(ref_dir)
            print(f" -> Reference Directory: {ref_dir} ({len(all_ref_files)} valid structure files detected)")
        elif target_ref.is_file():
            common_ref_file = target_ref
            print(f" -> Single Reference File: {common_ref_file.name} (common for all ligands)")
    
    # Gather and sort input ligand files by MW
    if args.ligand:
        if not args.ligand.is_file():
            print(f"❌ ERROR: Ligand file does not exist: {args.ligand}")
            sys.exit(1)
        ligand_files = [args.ligand]
    else:
        if not args.ligands.is_dir():
            print(f"❌ ERROR: Ligands directory does not exist: {args.ligands}")
            sys.exit(1)
        raw_files = [f for f in args.ligands.iterdir() if f.is_file() and f.suffix.lower() in {".pdb", ".mol2", ".sdf", ".sd"}]
        
        # Zero-ligand safety guard
        if not raw_files:
            print(f"❌ ERROR: No valid ligand files (.mol2, .sdf, .pdb) found in {args.ligands}!")
            sys.exit(1)

        print(" -> Sorting input ligands by Molecular Weight (MW)...")
        ligand_files = sorted(raw_files, key=lambda f: (get_ligand_molecular_weight(f), f.name))

    total_ligands = len(ligand_files)
    print("=" * 75)
    print("🚀 VADOCKER GPU-ACCELERATED LATTICE PIPELINE INITIALIZED")
    print(f" -> Mode          : {'HYDRATED (v5)' if args.hydrated else 'DRY DOCKING (v4)'}")
    print(f" -> Receptor      : {args.receptor.name} ({'Heme Metalloprotein' if is_heme_system else 'Non-Heme Globular Protein'})")
    print(f" -> Total Ligands : {total_ligands}")
    print(f" -> Grid Search   : Step {args.translation_step} Å | Rot {args.rotation_step}° SO(3)")
    print(f" -> Clash Filter  : Option B (Max clashes <= {args.max_clashes})")
    print(f" -> Sorting Order : Ascending Molecular Weight (MW)")
    if is_heme_system:
        print(f" -> Score Scheme  : Equal-Weight Composite for Unhindered Fe-N | Raw Score for others")
    else:
        print(f" -> Score Scheme  : Raw Score (Enhanced Gaussian VDW + H-bonds) across all ligands")
    if ref_dir:
        print(f" -> Reference Mode: Dynamic ligand matching from {ref_dir.name} (criteria: <lig>_lig_aln.* -> 4-letter code)")
    elif common_ref_file:
        print(f" -> Reference Mode: Single reference for all ligands ({common_ref_file.name})")
    else:
        print(f" -> Reference Mode: None specified -> Calculating RMSD w.r.t. Top Pose (ps001)")
    print(f" -> Output Path   : {args.out}")
    print("=" * 75)
    sys.stdout.flush()
    
    for idx, lf in enumerate(ligand_files, 1):
        ligand_name = lf.stem
        ligand_folder = Path(args.out) / ligand_name
        ligand_folder.mkdir(parents=True, exist_ok=True)
        
        ligand_mw = get_ligand_molecular_weight(lf)
        ligand_log_path = ligand_folder / f"{ligand_name}_docking.log"

        ref_file, curr_ref_atoms = resolve_ligand_reference(
            ref_dir=ref_dir,
            common_ref_file=common_ref_file,
            ligand_name=ligand_name,
            all_ref_files=all_ref_files
        )
        curr_ref_name = ref_file.name if ref_file else None

        ref_desc = curr_ref_name if curr_ref_name else "Top Pose (ps001)"
        print(f"\n[{idx}/{total_ligands}] >>> Processing: {ligand_name} (MW: {ligand_mw:.2f} Da | RMSD Ref: {ref_desc} | Log: {ligand_log_path.name})")
        sys.stdout.flush()

        with TeeLogger(ligand_log_path):
            t_lig_start = time.time()
            execute_single_docking(
                receptor_atoms=receptor_atoms,
                lig_path=lf,
                out_dir=args.out,
                center=tuple(args.center),
                size=args.size,
                step=args.translation_step,
                rotation_step=args.rotation_step,
                scoring_mode=args.scoring,
                top_poses=args.top_poses,
                ref_atoms=curr_ref_atoms,
                smiles_str=args.smiles,
                max_conformers=args.max_conformers,
                energy_window=args.energy_window,
                hydrated=args.hydrated,
                e_bulk=args.e_bulk,
                max_waters=args.max_waters
            )
                
            if ligand_folder.is_dir():
                process_and_redefine_workspace_scores(
                    output_dir=ligand_folder, 
                    receptor_atoms=receptor_atoms, 
                    ligand_file=lf, 
                    top_poses_limit=args.top_poses, 
                    max_clashes=args.max_clashes,
                    ref_atoms=curr_ref_atoms,
                    ref_name=curr_ref_name,
                    is_hydrated=args.hydrated
                )
            print(f"✓ Completed {ligand_name} (MW: {ligand_mw:.2f} Da) in {time.time() - t_lig_start:.2f}s")
            
        sys.stdout.flush()
            
    print("\n" + "=" * 75)
    print(f"🎉 All {total_ligands} GPU docking tasks completed successfully.")
    print("=" * 75)
    sys.stdout.flush()


if __name__ == "__main__":
    main()