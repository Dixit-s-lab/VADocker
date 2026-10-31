#!/usr/bin/env python3
import math
import heapq
import csv
import sys
import os
import time
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, replace
from itertools import product
from typing import Optional, List, Tuple, Set, Dict

# Ensure script directory is always in sys.path
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Force UTF-8 standard streams across Windows and Unix terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
from scipy.optimize import minimize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASS_BY_ELEMENT = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.904,
}

VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.40,
}

AMBER_PARAMS = {
    "C":     (1.9080, 0.0860, -0.15), "H":     (1.4870, 0.0157,  0.06),
    "O":     (1.6612, 0.2100, -0.45), "N":     (1.8240, 0.1700, -0.30),
    "S":     (2.0000, 0.2500,  0.05), "P":     (2.1000, 0.2000,  0.20),
    "F":     (1.7500, 0.0610, -0.20), "CL":    (1.9480, 0.2650, -0.15),
    "BR":    (2.2200, 0.3200, -0.10), "I":     (2.3500, 0.4000, -0.05),
    "FE1":   (1.2000, 0.0500,  2.00), "HM1_N": (1.8240, 0.1700, -0.25),
    "HM1_C": (1.9080, 0.0860,  0.05), "CM1_S": (2.0000, 0.2500, -0.60),
    "OW":    (1.7682, 0.1520, -0.834), "HW":   (0.0000, 0.0000,  0.417)
}

POLAR = {"N", "O", "S"}
HYDROPHOBIC = {"C", "CL", "BR", "I", "F"}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}

@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    res_name: str
    res_num: int
    chain: str
    element: str
    x: float
    y: float
    z: float
    r_min_half: float = 1.9080
    epsilon: float = 0.0860
    charge: float = 0.0000
    is_het_n: bool = False
    is_unhindered_n: bool = False
    is_aromatic: bool = False
    num_rot: int = 0
    lp_dx: float = 0.0
    lp_dy: float = 0.0
    lp_dz: float = 0.0


def is_fe_atom(atom: Atom) -> bool:
    """Detects catalytic Iron (Fe) in Heme or metalloprotein active sites."""
    if atom.element == "FE": return True
    if "FE" in atom.name.upper(): return True
    if atom.res_name.upper() in ("FE", "FE1", "HEM", "HEME") and "FE" in atom.name.upper():
        return True
    return False


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
        if res_name in {"HOH", "WAT", "TIP3", "H2O"}:
            if element == "O": r_h, eps, q = AMBER_PARAMS["OW"]
            elif element == "H": r_h, eps, q = AMBER_PARAMS["HW"]
        elif res_name == "FE1" or element == "FE": r_h, eps, q = AMBER_PARAMS["FE1"]
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


def parse_atoms_from_pdb_block(
    block: str, 
    het_n_indices: Optional[Set[int]] = None,
    unhindered_n_indices: Optional[Set[int]] = None,
    lp_vectors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    aromatic_indices: Optional[Set[int]] = None,
    num_rot: int = 0
) -> List[Atom]:
    atoms = []
    atom_idx = 0
    for line in block.splitlines():
        atom = parse_pdb_line(line)
        if atom:
            is_hn = bool(het_n_indices and atom_idx in het_n_indices)
            is_unh = bool(unhindered_n_indices and atom_idx in unhindered_n_indices)
            is_aro = bool(aromatic_indices and atom_idx in aromatic_indices)
            lp_dx, lp_dy, lp_dz = 0.0, 0.0, 0.0
            if is_hn and lp_vectors and atom_idx in lp_vectors:
                lp_dx, lp_dy, lp_dz = lp_vectors[atom_idx]
            atom = replace(
                atom, 
                is_het_n=is_hn, 
                is_unhindered_n=is_unh, 
                is_aromatic=is_aro,
                num_rot=num_rot,
                lp_dx=lp_dx, 
                lp_dy=lp_dy, 
                lp_dz=lp_dz
            )
            atoms.append(atom)
            atom_idx += 1
    return atoms


def separate_protein_and_waters(atoms: List[Atom]) -> Tuple[List[Atom], List[Atom]]:
    protein, waters = [], []
    for a in atoms:
        if a.res_name in {"HOH", "WAT", "TIP3", "H2O"}:
            waters.append(a)
        else:
            protein.append(a)
    return protein, waters


def _load_molecule_to_rdkit(ligand_file: Path, smiles_str: Optional[str] = None):
    from rdkit import Chem
    ext = ligand_file.suffix.lower()
    mol = None

    if smiles_str:
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol: mol = Chem.AddHs(mol)
        except Exception: mol = None

    if mol is None:
        if ext == ".mol2":
            try: mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False, sanitize=True)
            except Exception: mol = None
            if mol is None and shutil.which("obabel"):
                try:
                    res = subprocess.run(["obabel", "-imol2", str(ligand_file), "-osdf", "-h"], capture_output=True, text=True)
                    if res.returncode == 0 and res.stdout:
                        mol = Chem.MolFromMolBlock(res.stdout, removeHs=False)
                except Exception: mol = None
            if mol is None:
                try:
                    mol = Chem.MolFromMol2File(str(ligand_file), sanitize=False, removeHs=False)
                    if mol is not None: mol.UpdatePropertyCache(strict=False)
                except Exception: mol = None
        elif ext in (".sdf", ".sd"):
            try:
                suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
                if suppl and len(suppl) > 0: mol = suppl[0]
            except Exception: mol = None
        elif ext == ".pdb":
            try: mol = Chem.MolFromPDBFile(str(ligand_file), removeHs=False)
            except Exception: mol = None

    return mol


def analyze_heterocyclic_nitrogens(mol) -> Tuple[Set[int], Set[int]]:
    unhindered_n = set()
    hindered_n = set()
    if mol is None:
        return unhindered_n, hindered_n

    from rdkit import Chem
    smarts = "[$([n;X2]),$([#7;R;X2;!$([#7+]);!$([#7H])])]"
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return unhindered_n, hindered_n

    matches = mol.GetSubstructMatches(query)
    candidate_indices = {m[0] for m in matches}

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


def extract_nonterminal_rotatable_bonds(mol) -> List[Tuple[int, int, List[int]]]:
    if mol is None:
        return []
    from rdkit import Chem
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE: continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1: continue
        
        h_nbrs_1 = [n for n in a1.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != a2.GetIdx()]
        h_nbrs_2 = [n for n in a2.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != a1.GetIdx()]
        if len(h_nbrs_1) == 0 or len(h_nbrs_2) == 0:
            continue
            
        visited = {a1.GetIdx(), a2.GetIdx()}
        queue = [a2.GetIdx()]
        moving = [a2.GetIdx()]
        while queue:
            curr = queue.pop(0)
            for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx not in visited:
                    visited.add(n_idx)
                    moving.append(n_idx)
                    queue.append(n_idx)
        rot_bonds.append((a1.GetIdx(), a2.GetIdx(), moving))
    return rot_bonds


def generate_rdkit_conformers(
    ligand_file: Path,
    smiles_str: Optional[str] = None,
    max_conformers: int = 25,
    energy_window: float = 25.0,
    rmsd_prune: float = 0.5
) -> Tuple[List[List[Atom]], bool, bool, List[Tuple[int, int, List[int]]]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = _load_molecule_to_rdkit(ligand_file, smiles_str)
    if mol is None:
        fallback = read_pdb(ligand_file)
        return ([fallback] if fallback else []), False, False, []

    unhindered_n, hindered_n = analyze_heterocyclic_nitrogens(mol)

    try:
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
            unhindered_n, hindered_n = analyze_heterocyclic_nitrogens(mol)
    except Exception: pass

    het_n_indices = unhindered_n | hindered_n
    has_het_n = len(het_n_indices) > 0
    has_unhindered_n = len(unhindered_n) > 0

    try:
        aromatic_indices = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    except Exception:
        aromatic_indices = set()

    try: num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception: num_rot = 0

    rot_bonds_info = extract_nonterminal_rotatable_bonds(mol)

    num_to_embed = max(80, min(350, num_rot * 18))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.numThreads = 1

    cids = []
    try:
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_to_embed, params=params))
        if not cids:
            params.useRandomCoords = True
            params.maxAttempts = 1000
            cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_to_embed, params=params))
    except Exception: cids = []

    conformer_sets = []
    if cids:
        energy_list = []
        use_mmff = False
        try: use_mmff = AllChem.MMFFHasAllMoleculeProperties(mol)
        except Exception: use_mmff = False

        for cid in cids:
            try:
                if use_mmff:
                    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                    ff.Initialize(); ff.Minimize(maxIts=500)
                    e = ff.CalcEnergy()
                else:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                    ff.Initialize(); ff.Minimize(maxIts=500)
                    e = ff.CalcEnergy()
                energy_list.append((cid, e))
            except Exception: energy_list.append((cid, 0.0))

        min_e = min(e for _, e in energy_list)
        valid_confs = [(cid, e) for cid, e in energy_list if (e - min_e) <= energy_window]
        valid_confs.sort(key=lambda x: x[1])

        selected_cids = []
        for cid, _ in valid_confs:
            if len(selected_cids) >= max_conformers: break
            too_close = False
            for scid in selected_cids:
                try:
                    rmsd = AllChem.GetConformerRMS(mol, scid, cid, prealigned=False)
                    if rmsd < rmsd_prune:
                        too_close = True; break
                except Exception: pass
            if not too_close: selected_cids.append(cid)

        for cid in selected_cids:
            try:
                conf = mol.GetConformer(cid)
                lp_vectors = {}
                for n_idx in het_n_indices:
                    atom = mol.GetAtomWithIdx(n_idx)
                    nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                    if len(nbrs) == 2:
                        pos_n = np.array(conf.GetAtomPosition(n_idx))
                        pos_1 = np.array(conf.GetAtomPosition(nbrs[0]))
                        pos_2 = np.array(conf.GetAtomPosition(nbrs[1]))
                        v_lp = pos_n - 0.5 * (pos_1 + pos_2)
                        norm = np.linalg.norm(v_lp)
                        if norm > 1e-4:
                            lp_vectors[n_idx] = tuple(v_lp / norm)

                pdb_block = Chem.MolToPDBBlock(mol, confId=cid)
                atoms = parse_atoms_from_pdb_block(pdb_block, het_n_indices, unhindered_n, lp_vectors, aromatic_indices, num_rot)
                if atoms: conformer_sets.append(atoms)
            except Exception: pass

    if not conformer_sets:
        fallback = read_pdb(ligand_file)
        if fallback:
            conformer_sets.append([replace(a, num_rot=num_rot) for a in fallback])
    return conformer_sets, has_het_n, has_unhindered_n, rot_bonds_info


def get_so3_rotation_matrices(step_deg: float) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    eff_step = max(20.0, step_deg)
    n_points = min(max(16, int(900.0 / eff_step)), 64)
    n_roll = min(max(3, int(240.0 / eff_step)), 8)
    
    matrices = []
    golden = (1.0 + 5.0**0.5) / 2.0
    for i in range(n_points):
        z = 1.0 - (2.0 * i + 1.0) / n_points
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * i / golden
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        bx, by, bz = (1.0, 0.0, 0.0) if abs(x) < 0.8 else (0.0, 1.0, 0.0)
        dot = bx * x + by * y + bz * z
        ux, uy, uz = bx - dot * x, by - dot * y, bz - dot * z
        u_len = math.sqrt(ux*ux + uy*uy + uz*uz)
        ux, uy, uz = ux / u_len, uy / u_len, uz / u_len
        
        vx = y * uz - z * uy
        vy = z * ux - x * uz
        vz = x * uy - y * ux
        
        for j in range(n_roll):
            roll = 2.0 * math.pi * j / n_roll
            cr, sr = math.cos(roll), math.sin(roll)
            r_ux = cr * ux + sr * vx
            r_uy = cr * uy + sr * vy
            r_uz = cr * uz + sr * vz
            r_vx = -sr * ux + cr * vx
            r_vy = -sr * uy + cr * vy
            r_vz = -sr * uz + cr * vz
            matrices.append(((r_ux, r_vx, x), (r_uy, r_vy, y), (r_uz, r_vz, z)))
    return matrices


# =============================================================================
# CALIBRATED EMPIRICAL FREE ENERGY SCORING (RANGE: -12.0 to +12.0 kcal/mol)
# =============================================================================

def calculate_empirical_score(ligand_atoms: List[Atom], receptor_atoms: List[Atom]) -> Tuple[float, int, int, int]:
    base_score, clashes, contacts, hbonds = 0.0, 0, 0, 0
    fe_atoms = [r for r in receptor_atoms if is_fe_atom(r)]
    has_fe = len(fe_atoms) > 0
    has_het_n = any(getattr(a, "is_het_n", False) for a in ligand_atoms)
    enable_coord = has_fe and has_het_n
    coord_bonuses = []

    for l_at in ligand_atoms:
        if l_at.element == "H": continue
        l_rad = VDW_RADIUS.get(l_at.element, 1.70)
        is_hn = getattr(l_at, "is_het_n", False)
        is_unh = getattr(l_at, "is_unhindered_n", False)
        is_aro = getattr(l_at, "is_aromatic", False)

        for r_at in receptor_atoms:
            if r_at.element == "H": continue
            dx, dy, dz = l_at.x - r_at.x, l_at.y - r_at.y, l_at.z - r_at.z
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq > 25.0: continue
            
            is_fe = is_fe_atom(r_at)

            # Fe - Heterocyclic N coordination
            if is_hn and is_fe:
                dist = math.sqrt(dist_sq)
                clash_cutoff = 1.75 if is_unh else 1.90
                if dist < clash_cutoff:
                    clashes += 1
                    base_score += 3.5 * (clash_cutoff - dist + 0.05)
                elif clash_cutoff <= dist <= 5.0:
                    contacts += 1
                    un_fe = ((r_at.x - l_at.x) / dist, (r_at.y - l_at.y) / dist, (r_at.z - l_at.z) / dist)
                    cos_theta = l_at.lp_dx * un_fe[0] + l_at.lp_dy * un_fe[1] + l_at.lp_dz * un_fe[2]
                    
                    if cos_theta > 0.0:
                        f_ang = cos_theta * cos_theta
                        well_depth = -4.50 if is_unh else -1.80
                        sigma_sq = 1.445 if is_unh else 0.80
                        f_dist = math.exp(-((dist - 2.18) ** 2) / sigma_sq)
                        coord_bonuses.append(well_depth * f_dist * f_ang)
                continue

            # Standard Protein-Ligand, Water-Ligand, and Porphyrin contacts
            r_rad = VDW_RADIUS.get(r_at.element, 1.70)
            d_opt = l_rad + r_rad
            clash_dist = 0.72 * d_opt  # Softened to 0.72 to tolerate crystal packing compression
            
            if dist_sq < clash_dist * clash_dist:
                clashes += 1
                base_score += 3.0 * (clash_dist - math.sqrt(dist_sq) + 0.05)
            elif dist_sq <= 20.25:
                distance = math.sqrt(dist_sq)
                contacts += 1
                
                # 1. Soft-core Gaussian VDW attraction well
                diff = distance - d_opt
                base_score -= 0.040 * math.exp(-((diff / 0.60) ** 2))
                
                # 2. Hydrophobic contacts
                if l_at.element in HYDROPHOBIC and r_at.element in HYDROPHOBIC and 3.2 <= distance <= 4.2:
                    base_score -= 0.035

                # 3. Hydrogen bonding
                if l_at.element in POLAR and r_at.element in POLAR and 2.4 <= distance <= 3.4:
                    hbonds += 1
                    base_score -= 0.60

                # 4. Balanced aromatic pi-pi stacking with Heme and protein aromatic residues (PHE, TYR, TRP, HIS)
                r_res = r_at.res_name.upper()
                is_pi_target = (r_res in ("HEM", "HEME") or r_res in AROMATIC_RES) and (r_at.element.upper() in ("C", "N"))
                if is_aro and is_pi_target and 3.2 <= distance <= 4.5:
                    base_score -= 0.060 * math.exp(-(((distance - 3.65) / 0.45) ** 2))

                # 5. Distance-dependent Coulomb electrostatics
                if abs(l_at.charge * r_at.charge) > 1e-4:
                    r_sq_c = max(dist_sq, 4.0)
                    base_score += 0.08 * (83.0159 * l_at.charge * r_at.charge) / r_sq_c

    num_rot = getattr(ligand_atoms[0], "num_rot", 0) if ligand_atoms else 0
    rot_penalty = 0.20 * num_rot

    if enable_coord:
        best_coord = 0.0
        if coord_bonuses:
            # Exponential continuous attenuation replaces discrete 50% cut
            clash_scale = math.exp(-0.25 * clashes)
            best_coord = min(coord_bonuses) * clash_scale
        total_energy = base_score + best_coord + rot_penalty
    else:
        total_energy = base_score + rot_penalty

    total_raw_score = min(12.0, total_energy)
    return total_raw_score, clashes, contacts, hbonds


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
            if dist_sq < (0.72 * combined_rad) ** 2:
                if l_at.element != "H" and r_at.element != "H": clashes += 1
            
            dist = math.sqrt(dist_sq)
            if dist < 0.8: dist = 0.8
            r_min_ij = l_at.r_min_half + r_at.r_min_half
            eps_ij = math.sqrt(l_at.epsilon * r_at.epsilon)
            rm_over_r_6 = (r_min_ij / dist) ** 6
            total_vdw += eps_ij * (rm_over_r_6**2 - 2.0 * rm_over_r_6)
            total_elec += (l_at.charge * r_at.charge * COULOMB_CONSTANT) / (4.0 * dist_sq)
            
    scaled = (total_vdw + total_elec) * 0.20
    final_score = min(12.0, scaled)
    return final_score, clashes


def rodrigues_rotate_points(coords: np.ndarray, origin: np.ndarray, axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    v = coords - origin
    dot = np.sum(v * axis_unit, axis=1, keepdims=True)
    cross = np.cross(axis_unit, v)
    return origin + v * c + cross * s + axis_unit * dot * (1.0 - c)


def local_relax_rigid_pose(
    pose: List[Atom], 
    receptor_atoms: List[Atom], 
    max_iter: int = 45,
    rot_bonds_info: Optional[List[Tuple[int, int, List[int]]]] = None
) -> List[Atom]:
    if not pose: return pose
    
    cx = sum(a.x for a in pose) / len(pose)
    cy = sum(a.y for a in pose) / len(pose)
    cz = sum(a.z for a in pose) / len(pose)
    orig_coords = np.array([(a.x, a.y, a.z) for a in pose], dtype=np.float64)
    
    active_rot = rot_bonds_info[:5] if rot_bonds_info else []
    n_params = 6 + len(active_rot)
    
    def transform_coords(params):
        rigid_params = params[:6]
        torsion_params = params[6:]
        
        curr_coords = np.array(orig_coords, copy=True)
        if active_rot:
            for (a1_idx, a2_idx, moving_indices), d_phi_deg in zip(active_rot, torsion_params):
                if abs(d_phi_deg) < 1e-4: continue
                origin = curr_coords[a1_idx]
                axis = curr_coords[a2_idx] - origin
                norm = np.linalg.norm(axis)
                if norm < 1e-4: continue
                axis_unit = axis / norm
                m_list = list(moving_indices)
                curr_coords[m_list] = rodrigues_rotate_points(curr_coords[m_list], origin, axis_unit, math.radians(d_phi_deg))
        
        tx, ty, tz, rx, ry, rz = rigid_params
        ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
        cx_a, sx_a = math.cos(ax), math.sin(ax)
        cy_a, sy_a = math.cos(ay), math.sin(ay)
        cz_a, sz_a = math.cos(az), math.sin(az)
        
        r00 = cy_a * cz_a
        r01 = sx_a * sy_a * cz_a - cx_a * sz_a
        r02 = cx_a * sy_a * cz_a + sx_a * sz_a
        r10 = cy_a * sz_a
        r11 = sx_a * sy_a * sz_a + cx_a * cz_a
        r12 = cx_a * sy_a * sz_a - sx_a * cz_a
        r20 = -sy_a
        r21 = sx_a * cy_a
        r22 = cx_a * cy_a
        
        R = np.array([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]])
        
        diff = curr_coords - np.array([cx, cy, cz])
        transformed_pts = (diff @ R.T) + np.array([cx + tx, cy + ty, cz + tz])
        
        transformed_atoms = []
        for i, a in enumerate(pose):
            nx, ny, nz = transformed_pts[i]
            if a.is_het_n:
                nlp = R @ np.array([a.lp_dx, a.lp_dy, a.lp_dz])
                transformed_atoms.append(replace(a, x=float(nx), y=float(ny), z=float(nz), lp_dx=float(nlp[0]), lp_dy=float(nlp[1]), lp_dz=float(nlp[2])))
            else:
                transformed_atoms.append(replace(a, x=float(nx), y=float(ny), z=float(nz)))
        return transformed_atoms

    def objective(params):
        coords = transform_coords(params)
        score, _, _, _ = calculate_empirical_score(coords, receptor_atoms)
        return score

    # Calibrated Powell initial step direction bracket
    direc = np.eye(n_params)
    direc[0, 0] = 0.5   # 0.5 A tx
    direc[1, 1] = 0.5   # 0.5 A ty
    direc[2, 2] = 0.5   # 0.5 A tz
    direc[3, 3] = 10.0  # 10.0 deg rx
    direc[4, 4] = 10.0  # 10.0 deg ry
    direc[5, 5] = 10.0  # 10.0 deg rz
    for i in range(len(active_rot)):
        direc[6 + i, 6 + i] = 25.0  # 25.0 deg dihedral search step

    res = minimize(objective, [0.0]*n_params, method="Powell", options={"direc": direc, "maxiter": max_iter, "ftol": 0.001})
    if res.success or res.fun < objective([0.0]*n_params):
        return transform_coords(res.x)
    return pose


def parse_heavy_atom_coordinates_from_list(atom_list: List[Atom]) -> List[Tuple[float, float, float]]:
    return [(a.x, a.y, a.z) for a in atom_list if a.element != "H"]


def calculate_rmsd(coords_a: List[Tuple[float, float, float]], coords_b: List[Tuple[float, float, float]]) -> float:
    if not coords_a or not coords_b or len(coords_a) != len(coords_b): return float("inf")
    return math.sqrt(sum((xa - xb)**2 + (ya - yb)**2 + (za - zb)**2 for (xa, ya, za), (xb, yb, zb) in zip(coords_a, coords_b)) / len(coords_a))


def calculate_heavy_atom_rmsd(pose_atoms: List[Atom], reference_atoms: List[Atom]) -> float:
    pose_heavy = [a for a in pose_atoms if a.element != "H"]
    ref_heavy = [a for a in reference_atoms if a.element != "H"]
    if not pose_heavy or not ref_heavy: return 0.0
    
    sq_dist_sum = 0.0
    matched_ref_indices = set()
    for p_at in pose_heavy:
        min_sq_dist = float("inf")
        best_ref_idx = -1
        for idx, r_at in enumerate(ref_heavy):
            if idx in matched_ref_indices or r_at.element != p_at.element: continue
            d_sq = (p_at.x - r_at.x)**2 + (p_at.y - r_at.y)**2 + (p_at.z - r_at.z)**2
            if d_sq < min_sq_dist:
                min_sq_dist = d_sq
                best_ref_idx = idx
        if best_ref_idx != -1:
            sq_dist_sum += min_sq_dist
            matched_ref_indices.add(best_ref_idx)
        else:
            sq_dist_sum += 16.0
    return math.sqrt(sq_dist_sum / len(pose_heavy))


def calculate_bipartite_water_oxygen_rmsd(predicted_waters: List[List[Atom]], reference_waters: List[Atom]) -> Tuple[float, int, int]:
    pred_oxygens = [next(a for a in w if a.element == "O") for w in predicted_waters if any(a.element == "O" for a in w)]
    ref_oxygens = [a for a in reference_waters if a.element == "O"]
    
    if not pred_oxygens or not ref_oxygens:
        return 0.0, len(pred_oxygens), len(ref_oxygens)
        
    cost_matrix = []
    for p in pred_oxygens:
        row = []
        for r in ref_oxygens:
            row.append((p.x - r.x)**2 + (p.y - r.y)**2 + (p.z - r.z)**2)
        cost_matrix.append(row)
        
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_sq_dists = [cost_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    except Exception:
        matched_sq_dists = []
        used_cols = set()
        for r in range(len(cost_matrix)):
            best_c = -1
            best_val = float("inf")
            for c in range(len(cost_matrix[0])):
                if c not in used_cols and cost_matrix[r][c] < best_val:
                    best_val = cost_matrix[r][c]
                    best_c = c
            if best_c != -1:
                matched_sq_dists.append(best_val)
                used_cols.add(best_c)
                
    if not matched_sq_dists:
        return 0.0, len(pred_oxygens), len(ref_oxygens)
        
    rmsd = math.sqrt(sum(matched_sq_dists) / len(matched_sq_dists))
    return rmsd, len(pred_oxygens), len(ref_oxygens)


# =============================================================================
# GPU-ACCELERATED LATTICE BATCH SCORING KERNEL WITH TOP-K PRUNING
# =============================================================================

@torch.no_grad()
def evaluate_conformer_grid_on_gpu(
    lig_atoms: List[Atom],
    active_site: List[Atom],
    grid_points: List[Tuple[float, float, float]],
    rotation_matrices: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]],
    clash_limit: int,
    batch_size: int = 4096,
    k_per_batch: int = 30,
    max_candidates_per_conf: int = 150
) -> List[Tuple[float, int, int, int, List[Atom]]]:
    lig_heavy = [a for a in lig_atoms if a.element != "H"]
    rec_heavy = [a for a in active_site if a.element != "H"]
    
    lx = [a.x for a in lig_atoms]; ly = [a.y for a in lig_atoms]; lz = [a.z for a in lig_atoms]
    cx, cy, cz = sum(lx)/len(lx), sum(ly)/len(ly), sum(lz)/len(lz)
    
    lig_coords_centered = torch.tensor([[a.x - cx, a.y - cy, a.z - cz] for a in lig_heavy], dtype=torch.float32, device=DEVICE)
    lig_all_centered = torch.tensor([[a.x - cx, a.y - cy, a.z - cz] for a in lig_atoms], dtype=torch.float32, device=DEVICE)
    rec_coords = torch.tensor([[a.x, a.y, a.z] for a in rec_heavy], dtype=torch.float32, device=DEVICE)
    
    lig_rad = torch.tensor([VDW_RADIUS.get(a.element, 1.70) for a in lig_heavy], dtype=torch.float32, device=DEVICE)
    rec_rad = torch.tensor([VDW_RADIUS.get(a.element, 1.70) for a in rec_heavy], dtype=torch.float32, device=DEVICE)
    
    lig_is_polar = torch.tensor([a.element in POLAR for a in lig_heavy], dtype=torch.bool, device=DEVICE)
    rec_is_polar = torch.tensor([a.element in POLAR for a in rec_heavy], dtype=torch.bool, device=DEVICE)
    
    lig_is_hphobe = torch.tensor([a.element in HYDROPHOBIC for a in lig_heavy], dtype=torch.bool, device=DEVICE)
    rec_is_hphobe = torch.tensor([a.element in HYDROPHOBIC for a in rec_heavy], dtype=torch.bool, device=DEVICE)
    
    lig_is_het_n = torch.tensor([getattr(a, "is_het_n", False) for a in lig_heavy], dtype=torch.bool, device=DEVICE)
    lig_is_unh_n = torch.tensor([getattr(a, "is_unhindered_n", False) for a in lig_heavy], dtype=torch.bool, device=DEVICE)
    rec_is_fe = torch.tensor([is_fe_atom(a) for a in rec_heavy], dtype=torch.bool, device=DEVICE)
    enable_coord = lig_is_het_n.any().item() and rec_is_fe.any().item()
    
    lig_is_aromatic = torch.tensor([getattr(a, "is_aromatic", False) for a in lig_heavy], dtype=torch.bool, device=DEVICE)
    rec_is_aromatic = torch.tensor([
        ((getattr(a, "res_name", "").upper() in ("HEM", "HEME") or getattr(a, "res_name", "").upper() in AROMATIC_RES)
         and a.element.upper() in ("C", "N"))
        for a in rec_heavy
    ], dtype=torch.bool, device=DEVICE)
    pair_is_pipi = lig_is_aromatic.unsqueeze(1) & rec_is_aromatic.unsqueeze(0)

    # Electrostatic partial charges
    lig_charge = torch.tensor([a.charge for a in lig_heavy], dtype=torch.float32, device=DEVICE)
    rec_charge = torch.tensor([a.charge for a in rec_heavy], dtype=torch.float32, device=DEVICE)
    pair_charge = lig_charge.unsqueeze(1) * rec_charge.unsqueeze(0)
    has_charges = (torch.abs(pair_charge) > 1e-4).any().item()

    if enable_coord:
        het_indices = torch.where(lig_is_het_n)[0]
        fe_idx = torch.where(rec_is_fe)[0][0]
        fe_pos = rec_coords[fe_idx]
        lp_vecs_unrot = torch.tensor([[a.lp_dx, a.lp_dy, a.lp_dz] for a in lig_heavy], dtype=torch.float32, device=DEVICE)
        pair_is_coord = lig_is_het_n.unsqueeze(1) & rec_is_fe.unsqueeze(0)
    else:
        pair_is_coord = torch.zeros((len(lig_heavy), len(rec_heavy)), dtype=torch.bool, device=DEVICE)

    R_comb = 0.72 * (lig_rad.unsqueeze(1) + rec_rad.unsqueeze(0))
    if enable_coord:
        clash_cutoffs = torch.where(lig_is_unh_n, torch.tensor(1.75, device=DEVICE), torch.tensor(1.90, device=DEVICE))
        R_comb = torch.where(pair_is_coord, clash_cutoffs.unsqueeze(1), R_comb)
    
    R_comb_sq = R_comb ** 2
    D_opt = (lig_rad.unsqueeze(1) + rec_rad.unsqueeze(0)).unsqueeze(0)

    pair_is_polar = lig_is_polar.unsqueeze(1) & rec_is_polar.unsqueeze(0)
    pair_is_hphobe = lig_is_hphobe.unsqueeze(1) & rec_is_hphobe.unsqueeze(0)
    
    y2 = (rec_coords ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)
    
    rot_tensors = torch.tensor(rotation_matrices, dtype=torch.float32, device=DEVICE)
    trans_tensors = torch.tensor(grid_points, dtype=torch.float32, device=DEVICE)
    
    n_rot = len(rot_tensors)
    n_trans = len(trans_tensors)
    total_transforms = n_trans * n_rot
    
    batch_candidates = []
    
    for start_idx in range(0, total_transforms, batch_size):
        end_idx = min(start_idx + batch_size, total_transforms)
        b_size = end_idx - start_idx
        
        indices = torch.arange(start_idx, end_idx, device=DEVICE)
        t_indices = torch.div(indices, n_rot, rounding_mode="floor")
        r_indices = indices % n_rot
        
        sub_rot = rot_tensors[r_indices]
        sub_trans = trans_tensors[t_indices]
        
        coords_batch = torch.bmm(
            lig_coords_centered.unsqueeze(0).expand(b_size, -1, -1),
            sub_rot.transpose(1, 2)
        ) + sub_trans.unsqueeze(1)
        
        x2 = (coords_batch ** 2).sum(dim=-1, keepdim=True)
        xy = torch.matmul(coords_batch, rec_coords.T)
        dist_sq = torch.clamp(x2 + y2 - 2.0 * xy, min=0.0)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        clash_mask = dist_sq < R_comb_sq.unsqueeze(0)
        clashes = clash_mask.sum(dim=(1, 2))
        valid_b = clashes <= clash_limit
        
        if not valid_b.any():
            continue
            
        clash_penalties = torch.where(clash_mask, 3.0 * (R_comb.unsqueeze(0) - dist + 0.05), torch.zeros_like(dist))
        
        # 1. Soft-core Gaussian VDW attraction (evaluated within 4.5 A)
        contact_mask = (dist_sq >= R_comb_sq.unsqueeze(0)) & (dist_sq <= 20.25)
        vdw_diff = dist - D_opt
        gauss_vdw_rewards = torch.where(contact_mask, -0.040 * torch.exp(-((vdw_diff / 0.60) ** 2)), torch.zeros_like(dist))
        
        # 2. Hydrophobic contacts
        hphobe_mask = (dist >= 3.2) & (dist <= 4.2) & pair_is_hphobe.unsqueeze(0)
        hphobe_rewards = torch.where(hphobe_mask, -0.035, torch.zeros_like(dist))

        # 3. Hydrogen bonding
        hbond_mask = (dist >= 2.4) & (dist <= 3.4) & pair_is_polar.unsqueeze(0)
        hbond_rewards = torch.where(hbond_mask, -0.60, torch.zeros_like(dist))

        # 4. Balanced aromatic pi-pi stacking with Heme and protein aromatic residues
        pipi_mask = (dist >= 3.2) & (dist <= 4.5) & pair_is_pipi.unsqueeze(0)
        pipi_rewards = torch.where(pipi_mask, -0.060 * torch.exp(-(((dist - 3.65) / 0.45) ** 2)), torch.zeros_like(dist))

        # 5. Distance-dependent Coulomb electrostatics
        if has_charges:
            r_sq_c = torch.clamp(dist_sq, min=4.0)
            elec_rewards = (0.08 * 83.0159 * pair_charge.unsqueeze(0)) / r_sq_c
            total_elec = elec_rewards.sum(dim=(1, 2))
        else:
            total_elec = torch.zeros(b_size, device=DEVICE)
        
        base_score = (clash_penalties + gauss_vdw_rewards + hphobe_rewards + hbond_rewards + pipi_rewards).sum(dim=(1, 2)) + total_elec
        hbonds = hbond_mask.sum(dim=(1, 2))
        contacts = contact_mask.sum(dim=(1, 2))

        num_rot = getattr(lig_atoms[0], "num_rot", 0) if lig_atoms else 0
        rot_penalty = 0.20 * num_rot
        
        if enable_coord:
            lp_rot_batch = torch.bmm(
                lp_vecs_unrot.unsqueeze(0).expand(b_size, -1, -1),
                sub_rot.transpose(1, 2)
            )
            
            pos_n = coords_batch[:, het_indices, :]
            u_lp = lp_rot_batch[:, het_indices, :]
            
            vec_n_fe = fe_pos.unsqueeze(0).unsqueeze(0) - pos_n
            dist_n_fe = torch.sqrt((vec_n_fe ** 2).sum(dim=-1) + 1e-8)
            u_n_fe = vec_n_fe / dist_n_fe.unsqueeze(-1)
            
            cos_theta = (u_lp * u_n_fe).sum(dim=-1)
            f_ang = torch.clamp(cos_theta, min=0.0) ** 2
            
            is_unh_subset = lig_is_unh_n[het_indices].unsqueeze(0)
            well_depth = torch.where(is_unh_subset, torch.tensor(-4.50, device=DEVICE), torch.tensor(-1.80, device=DEVICE))
            sigma_sq = torch.where(is_unh_subset, torch.tensor(1.445, device=DEVICE), torch.tensor(0.80, device=DEVICE))
            
            f_dist = torch.exp(-((dist_n_fe - 2.18) ** 2) / sigma_sq)
            dist_valid = (dist_n_fe >= 1.75) & (dist_n_fe <= 5.0)
            f_dist = torch.where(dist_valid, f_dist, torch.zeros_like(f_dist))
            
            clash_scale = torch.exp(-0.25 * clashes.float()).unsqueeze(1)
            rewards_k = well_depth * f_dist * f_ang * clash_scale
            coord_bonus = rewards_k.min(dim=1)[0]
            
            scores = torch.clamp_max(base_score + coord_bonus + rot_penalty, max=12.0)
        else:
            scores = torch.clamp_max(base_score + rot_penalty, max=12.0)
        
        # GPU Top-K Pruning: Extract only top-k candidates per batch to eliminate CPU object creation bottleneck
        masked_scores = torch.where(valid_b, scores, torch.tensor(1e6, device=DEVICE))
        k = min(k_per_batch, b_size)
        top_vals, top_indices = torch.topk(masked_scores, k=k, largest=False)
        
        valid_k = top_vals < 1e5
        if valid_k.any():
            good_vals = top_vals[valid_k].cpu().numpy()
            good_idx = top_indices[valid_k].cpu().numpy()
            g_indices = start_idx + good_idx
            cls_cpu = clashes[good_idx].cpu().numpy()
            cnt_cpu = contacts[good_idx].cpu().numpy()
            hb_cpu = hbonds[good_idx].cpu().numpy()
            
            for i in range(len(good_vals)):
                batch_candidates.append((float(good_vals[i]), int(cls_cpu[i]), int(cnt_cpu[i]), int(hb_cpu[i]), int(g_indices[i])))

    if not batch_candidates:
        return []

    batch_candidates.sort(key=lambda x: x[0])
    top_candidates = batch_candidates[:max_candidates_per_conf]

    # Batch reconstruct coordinates on GPU for only the top winning transforms
    top_g_indices = torch.tensor([c[4] for c in top_candidates], device=DEVICE)
    top_t_indices = torch.div(top_g_indices, n_rot, rounding_mode="floor")
    top_r_indices = top_g_indices % n_rot

    sub_rot = rot_tensors[top_r_indices]
    sub_trans = trans_tensors[top_t_indices]

    sub_all_batch = torch.bmm(
        lig_all_centered.unsqueeze(0).expand(len(top_candidates), -1, -1),
        sub_rot.transpose(1, 2)
    ) + sub_trans.unsqueeze(1)

    sub_all_cpu = sub_all_batch.cpu().numpy()
    
    surviving_poses = []
    for k in range(len(top_candidates)):
        c_mat = sub_all_cpu[k]
        constructed_pose = [replace(lig_atoms[ai], x=float(c_mat[ai, 0]), y=float(c_mat[ai, 1]), z=float(c_mat[ai, 2])) for ai in range(len(lig_atoms))]
        surviving_poses.append((top_candidates[k][0], top_candidates[k][1], top_candidates[k][2], top_candidates[k][3], constructed_pose))

    return surviving_poses


# =============================================================================
# GPU ELECTROSTATIC POTENTIAL & FIELD SOLVATION
# =============================================================================

@torch.no_grad()
def compute_field_grid_gpu(grid_points: List[Tuple[float, float, float]], source_atoms: List[Atom]) -> Tuple[torch.Tensor, torch.Tensor]:
    charged_atoms = [a for a in source_atoms if abs(a.charge) > 1e-4]
    if not charged_atoms:
        return torch.zeros(len(grid_points)), torch.zeros((len(grid_points), 3))
        
    g_pts = torch.tensor(grid_points, dtype=torch.float32, device=DEVICE)
    s_pts = torch.tensor([[a.x, a.y, a.z] for a in charged_atoms], dtype=torch.float32, device=DEVICE)
    q_vals = torch.tensor([a.charge for a in charged_atoms], dtype=torch.float32, device=DEVICE)
    
    diff = g_pts.unsqueeze(1) - s_pts.unsqueeze(0)
    r_sq = torch.clamp((diff ** 2).sum(dim=-1), min=0.25)
    r = torch.sqrt(r_sq)
    
    eps_r = 4.0 * r
    coulomb = 332.0636
    
    pot_matrix = (coulomb * q_vals.unsqueeze(0)) / (eps_r * r)
    potentials = pot_matrix.sum(dim=1).cpu()
    
    f_coeff = (coulomb * q_vals.unsqueeze(0)) / (eps_r * r_sq * r)
    e_fields = (f_coeff.unsqueeze(-1) * diff).sum(dim=1).cpu()
    
    return potentials, e_fields


def generate_tip3p_water_atoms(ox: float, oy: float, oz: float, field_vec: Tuple[float, float, float], start_serial: int = 5000) -> List[Atom]:
    fx, fy, fz = field_vec
    norm = math.sqrt(fx*fx + fy*fy + fz*fz)
    ux, uy, uz = (fx/norm, fy/norm, fz/norm) if norm >= 1e-4 else (0.0, 0.0, 1.0)
    
    vx, vy, vz = (1.0, 0.0, 0.0) if abs(ux) < 0.8 else (0.0, 1.0, 0.0)
    proj = vx*ux + vy*uy + vz*uz
    wx, wy, wz = vx - proj*ux, vy - proj*uy, vz - proj*uz
    w_norm = math.sqrt(wx*wx + wy*wy + wz*wz)
    wx, wy, wz = wx / w_norm, wy / w_norm, wz / w_norm
    
    r_oh = 0.9572
    half_angle = math.radians(104.52 / 2.0)
    cos_a, sin_a = math.cos(half_angle), math.sin(half_angle)
    
    h1_x = ox + r_oh * (cos_a * ux + sin_a * wx)
    h1_y = oy + r_oh * (cos_a * uy + sin_a * wy)
    h1_z = oz + r_oh * (cos_a * uz + sin_a * wz)
    h2_x = ox + r_oh * (cos_a * ux - sin_a * wx)
    h2_y = oy + r_oh * (cos_a * uy - sin_a * wy)
    h2_z = oz + r_oh * (cos_a * uz - sin_a * wz)
    
    return [
        Atom(start_serial, "O", "HOH", 1, "W", "O", ox, oy, oz, AMBER_PARAMS["OW"][0], AMBER_PARAMS["OW"][1], AMBER_PARAMS["OW"][2]),
        Atom(start_serial + 1, "H1", "HOH", 1, "W", "H", h1_x, h1_y, h1_z, AMBER_PARAMS["HW"][0], AMBER_PARAMS["HW"][1], AMBER_PARAMS["HW"][2]),
        Atom(start_serial + 2, "H2", "HOH", 1, "W", "H", h2_x, h2_y, h2_z, AMBER_PARAMS["HW"][0], AMBER_PARAMS["HW"][1], AMBER_PARAMS["HW"][2])
    ]


def place_and_optimize_pocket_waters_gpu(
    docked_ligand: List[Atom],
    active_site_atoms: List[Atom],
    center: tuple,
    size: float,
    e_bulk: float = -9.5,
    max_waters: int = 10,
    grid_spacing: float = 1.0
) -> List[List[Atom]]:
    env_atoms = active_site_atoms + docked_ligand
    margin = size / 2.0
    
    x_range = [center[0] - margin + i * grid_spacing for i in range(int(size / grid_spacing) + 1)]
    y_range = [center[1] - margin + i * grid_spacing for i in range(int(size / grid_spacing) + 1)]
    z_range = [center[2] - margin + i * grid_spacing for i in range(int(size / grid_spacing) + 1)]
    
    void_points = []
    for gx, gy, gz in product(x_range, y_range, z_range):
        min_lig_d = min(math.sqrt((gx - a.x)**2 + (gy - a.y)**2 + (gz - a.z)**2) for a in docked_ligand if a.element != "H")
        if not (2.2 <= min_lig_d <= 4.2): continue
        if any((gx - a.x)**2 + (gy - a.y)**2 + (gz - a.z)**2 < (0.75 * (1.52 + VDW_RADIUS.get(a.element, 1.70)))**2 for a in env_atoms if a.element != "H"):
            continue
        void_points.append((gx, gy, gz))
        
    if not void_points: return []
        
    pots, fields = compute_field_grid_gpu(void_points, env_atoms)
    candidate_seeds = [(float(pots[i]), void_points[i], (float(fields[i,0]), float(fields[i,1]), float(fields[i,2]))) for i in range(len(void_points))]
    candidate_seeds.sort(key=lambda x: x[0])
    
    retained_waters = []
    occupied_centers = []
    
    for _, (sx, sy, sz), e_field in candidate_seeds:
        if len(retained_waters) >= max_waters: break
        if any(math.sqrt((sx - ox)**2 + (sy - oy)**2 + (sz - oz)**2) < 2.6 for ox, oy, oz in occupied_centers): continue
        
        water_mol = generate_tip3p_water_atoms(sx, sy, sz, e_field, start_serial=5000 + len(retained_waters)*3)
        emp_s, _, _, _ = calculate_empirical_score(water_mol, env_atoms)
        amb_s, _ = calculate_amber_score(water_mol, env_atoms)
        e_int = emp_s + amb_s
        
        if e_int <= e_bulk:
            retained_waters.append(water_mol)
            occupied_centers.append((sx, sy, sz))
            
    return retained_waters


# =============================================================================
# MASTER UNIFIED DOCKING DRIVER (DRY & HYDRATED SUPPORT)
# =============================================================================

def execute_single_docking(
    receptor_atoms: List[Atom], 
    lig_path: Path, 
    out_dir: Path, 
    center: tuple, 
    size: float, 
    step: float, 
    rotation_step: float, 
    scoring_mode: str, 
    top_poses: int, 
    ref_atoms: Optional[List[Atom]] = None, 
    smiles_str: Optional[str] = None,
    max_conformers: int = 25,
    energy_window: float = 25.0,
    hydrated: Optional[bool] = None,
    e_bulk: float = -9.5,
    max_waters: int = 10,
    **kwargs
) -> str:
    if isinstance(hydrated, (float, int)) and not isinstance(hydrated, bool):
        passed_e_bulk = float(hydrated)
        passed_max_waters = int(e_bulk) if isinstance(e_bulk, (int, float)) else 10
        e_bulk = passed_e_bulk
        max_waters = passed_max_waters
        is_hydrated_mode = kwargs.get("hydrated", True)
    elif hydrated is None:
        is_hydrated_mode = kwargs.get("hydrated", False)
    else:
        is_hydrated_mode = bool(hydrated)

    t_start = time.time()
    ligand_base_name = lig_path.stem
    target_workspace = out_dir / ligand_base_name
    target_workspace.mkdir(parents=True, exist_ok=True)
    
    rec_protein, rec_waters = separate_protein_and_waters(receptor_atoms)
    ref_ligand_atoms, ref_water_atoms = separate_protein_and_waters(ref_atoms) if ref_atoms else ([], [])
    
    mode_name = "HYDRATED (v5)" if is_hydrated_mode else "DRY DOCKING (v4)"
    print(f"\n⚡ Processing Molecule Target ({mode_name}): {lig_path.name} [Device: {DEVICE.type.upper()}]")
    
    # --- Step 1: Conformer Generation with Heterocycle & Rotatable Bond Extraction ---
    t_s1 = time.time()
    step_total = "7" if is_hydrated_mode else "5"
    print(f" -> [1/{step_total}] Sampling RDKit rotatable conformer ensemble (ETKDGv3, {energy_window:.1f} kcal/mol window)...")
    conformer_atom_sets, has_het_n, has_unhindered_n, rot_bonds_info = generate_rdkit_conformers(
        lig_path, smiles_str, max_conformers=max_conformers, energy_window=energy_window, rmsd_prune=0.5
    )
    print(f"    | Generated {len(conformer_atom_sets)} stable structures (Active Non-Terminal Torsions: {len(rot_bonds_info[:5])}) in {time.time() - t_s1:.2f}s")
    if has_unhindered_n:
        print("    | [CHEM] UNHINDERED heterocyclic nitrogen detected -> High-affinity Fe-coordination active.")
    elif has_het_n:
        print("    | [CHEM] HINDERED (ortho-substituted) heterocyclic nitrogen detected -> Standard scheme adopted.")
    else:
        print("    | [CHEM] Non-coordinating ligand -> Enhanced Gaussian VDW packing, Electrostatics & Aromatic Stacking active.")
    
    # --- Step 2: Pocket Geometry Isolation ---
    t_s2 = time.time()
    print(f" -> [2/{step_total}] Mapping active site pocket boundary matrix...")
    margin = size / 2.0
    clash_margin = margin + 5.0
    active_site = [a for a in rec_protein if 
                   center[0]-clash_margin <= a.x <= center[0]+clash_margin and
                   center[1]-clash_margin <= a.y <= center[1]+clash_margin and
                   center[2]-clash_margin <= a.z <= center[2]+clash_margin]
    
    fe_atoms = [a for a in active_site if is_fe_atom(a)]
    if fe_atoms:
        print(f"    | Catalytic Iron identified in active site at ({fe_atoms[0].x:.2f}, {fe_atoms[0].y:.2f}, {fe_atoms[0].z:.2f})")

    x_range = [center[0] - margin + i*step for i in range(int(size/step)+1)]
    y_range = [center[1] - margin + i*step for i in range(int(size/step)+1)]
    z_range = [center[2] - margin + i*step for i in range(int(size/step)+1)]
    
    grid_combinations = []
    for gx, gy, gz in product(x_range, y_range, z_range):
        too_close = False
        for r_at in active_site:
            if r_at.element == "H": continue
            if (gx - r_at.x)**2 + (gy - r_at.y)**2 + (gz - r_at.z)**2 < 4.0:
                too_close = True; break
        if not too_close: grid_combinations.append((gx, gy, gz))
    print(f"    | Active search nodes: {len(grid_combinations)} in {time.time() - t_s2:.2f}s")
    
    # --- Step 3: GPU Batch Transformation with Top-K Filtering ---
    t_s3 = time.time()
    print(f" -> [3/{step_total}] Running GPU batch transformations (SO(3) Rotations, Batch: 4096, Top-K per Batch)...")
    rotation_matrices = get_so3_rotation_matrices(rotation_step)
    num_heavy = len([a for a in conformer_atom_sets[0] if a.element != "H"])
    clash_limit = max(8, int(0.20 * num_heavy))
    
    raw_heap = []
    pose_counter = 0
    for lig_atoms in conformer_atom_sets:
        survivors = evaluate_conformer_grid_on_gpu(
            lig_atoms, active_site, grid_combinations, rotation_matrices, clash_limit, batch_size=4096, k_per_batch=30, max_candidates_per_conf=150
        )
        for score, cls, cnt, hb, pose in survivors:
            heapq.heappush(raw_heap, (score, cls, cnt, hb, pose_counter, pose))
            pose_counter += 1
    print(f"    | GPU lattice search completed: {len(raw_heap)} highly viable candidates in {time.time() - t_s3:.2f}s")
    
    # --- Step 4: Relax-First, Cluster-Second Strategy with Bracketed Powell Minimization ---
    t_s4 = time.time()
    relax_iters = 45
    print(f" -> [4/{step_total}] Executing Powell 6-DOF + Torsion line-search minimization & cluster pruning...")
    
    max_seeds = max(90, top_poses * 9)
    seed_candidates = []
    while raw_heap and len(seed_candidates) < max_seeds:
        score, cls, cnt, hb, _, pose = heapq.heappop(raw_heap)
        coords = parse_heavy_atom_coordinates_from_list(pose)
        if not any(calculate_rmsd(coords, s_coords) < 0.8 for _, _, _, _, s_coords, _ in seed_candidates):
            seed_candidates.append((score, cls, cnt, hb, coords, pose))
            
    relaxed_pool = []
    for _, _, _, _, _, pose in seed_candidates:
        relaxed_pose = local_relax_rigid_pose(pose, active_site, max_iter=relax_iters, rot_bonds_info=rot_bonds_info)
        r_score, r_cls, r_cnt, r_hb = calculate_empirical_score(relaxed_pose, active_site)
        r_coords = parse_heavy_atom_coordinates_from_list(relaxed_pose)
        relaxed_pool.append((r_score, r_cls, r_cnt, r_hb, relaxed_pose, r_coords))
        
    relaxed_pool.sort(key=lambda x: x[0])
    
    unique_candidates = []
    for cand in relaxed_pool:
        r_coords = cand[5]
        if not any(calculate_rmsd(r_coords, u[5]) < 1.5 for u in unique_candidates):
            unique_candidates.append(cand)
            if len(unique_candidates) >= top_poses:
                break
                
    print(f"    | Refined {len(unique_candidates)} unique minima in {time.time() - t_s4:.2f}s")
    
    csv_out = target_workspace / "scores.csv"

    # --- Mode Branching: Hydrated vs. Dry Docking ---
    if is_hydrated_mode:
        t_s5 = time.time()
        print(f" -> [5/7] Computing GPU electrostatic vector fields & seeding field-aligned waters...")
        hydrated_poses = []
        for score, cls, cnt, hb, pose, coords in unique_candidates:
            placed_waters = place_and_optimize_pocket_waters_gpu(
                docked_ligand=pose, active_site_atoms=active_site, center=center,
                size=size, e_bulk=e_bulk, max_waters=max_waters
            )
            hydrated_poses.append((score, cls, cnt, hb, pose, placed_waters))
        print(f"    | Field placement completed in {time.time() - t_s5:.2f}s")
        
        t_s6 = time.time()
        print(f" -> [6/7] Running dual RMSD verification...")
        eval_results = []
        for score, cls, cnt, hb, pose, waters in hydrated_poses:
            target_ref = ref_ligand_atoms if ref_ligand_atoms else (ref_atoms if ref_atoms else None)
            lig_rmsd = calculate_heavy_atom_rmsd(pose, target_ref) if target_ref else 0.0
            w_rmsd, n_pred_w, n_ref_w = calculate_bipartite_water_oxygen_rmsd(waters, ref_water_atoms)
            eval_results.append({
                "raw_score": score, "clashes": cls, "contacts": cnt, "hbonds": hb,
                "pose": pose, "waters": waters, "rmsd_ligand": lig_rmsd,
                "rmsd_water_o": w_rmsd, "retained_waters": n_pred_w, "ref_waters": n_ref_w
            })
        print(f"    | Dual verification completed in {time.time() - t_s6:.2f}s")
        
        t_s7 = time.time()
        print(f" -> [7/7] Exporting composite complexes and ranking matrices...")
        with csv_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pose_rank", "score", "raw_score", "clashes", "contacts", "hbonds", "rmsd_to_reference", "rmsd_water_o", "retained_waters", "output_file"])
            for rank, res in enumerate(eval_results, 1):
                pose_fn = f"{ligand_base_name}_ps{rank:03d}.pdb"
                pose_file = target_workspace / pose_fn
                lines = [f"HETATM{a.serial:>5} {a.name:<4} {a.res_name:<3} {a.chain}{a.res_num:>4}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00           {a.element:<2}" for a in res["pose"]]
                for w in res["waters"]:
                    for wa in w:
                        lines.append(f"HETATM{wa.serial:>5} {wa.name:<4} {wa.res_name:<3} {wa.chain}{wa.res_num:>4}    {wa.x:8.3f}{wa.y:8.3f}{wa.z:8.3f}  1.00  0.00           {wa.element:<2}")
                pose_file.write_text("\n".join(lines) + "\nEND\n")
                writer.writerow([rank, f"{res['raw_score']:.4f}", f"{res['raw_score']:.4f}", res["clashes"], res["contacts"], res["hbonds"], f"{res['rmsd_ligand']:.4f}", f"{res['rmsd_water_o']:.4f}", res["retained_waters"], pose_fn])
                
        print("="*55)
        print(f"📊 HYDRATED DOCKING SUMMARY: {lig_path.name}")
        print("="*55)
        if eval_results:
            top = eval_results[0]
            print(f"  • Best Docking Score (ΔG)  : {top['raw_score']:.4f} kcal/mol")
            if fe_atoms and has_het_n:
                fe_a = fe_atoms[0]
                het_details = []
                for a in top["pose"]:
                    if getattr(a, "is_het_n", False):
                        dist = math.sqrt((a.x - fe_a.x)**2 + (a.y - fe_a.y)**2 + (a.z - fe_a.z)**2)
                        un_fe = ((fe_a.x - a.x)/dist, (fe_a.y - a.y)/dist, (fe_a.z - a.z)/dist)
                        cos_theta = a.lp_dx * un_fe[0] + a.lp_dy * un_fe[1] + a.lp_dz * un_fe[2]
                        het_details.append((dist, cos_theta, getattr(a, "is_unhindered_n", False)))
                if het_details:
                    min_dist, best_cos, is_unh = min(het_details, key=lambda x: x[0])
                    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, best_cos))))
                    tag = "Unhindered" if is_unh else "Hindered"
                    print(f"  • Top Pose Fe-N Distance   : {min_dist:.3f} Å ({tag}, Vector Alignment: {angle_deg:.1f}° off Fe-axis)")
            if ref_ligand_atoms or ref_atoms:
                print(f"  • Top Pose Ligand RMSD     : {top['rmsd_ligand']:.4f} Å")
            print(f"  • Water Oxygen RMSD        : {top['rmsd_water_o']:.4f} Å")
            print(f"  • Retained Favorable Waters: {top['retained_waters']} molecules")

    else:
        t_s5 = time.time()
        print(f" -> [5/5] Exporting target outputs...")
        with csv_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pose_rank", "score", "raw_score", "clashes", "contacts", "hbonds", "rmsd_to_reference", "output_file"])
            for rank, (score, cls, cnt, hb, pose, pose_coords) in enumerate(unique_candidates, 1):
                target_ref = ref_ligand_atoms if ref_ligand_atoms else (ref_atoms if ref_atoms else None)
                ref_rmsd = calculate_heavy_atom_rmsd(pose, target_ref) if target_ref else 0.0
                pose_fn = f"{ligand_base_name}_ps{rank:03d}.pdb"
                pose_file = target_workspace / pose_fn
                lines = [f"HETATM{a.serial:>5} {a.name:<4} {a.res_name:<3} {a.chain}{a.res_num:>4}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00           {a.element:<2}" for a in pose]
                pose_file.write_text("\n".join(lines) + "\nEND\n")
                writer.writerow([rank, f"{score:.4f}", f"{score:.4f}", cls, cnt, hb, f"{ref_rmsd:.4f}", pose_fn])
                
        print("="*50)
        print(f"📊 DOCKING RESULTS SUMMARY: {lig_path.name}")
        print("="*50)
        if unique_candidates:
            best_pose = unique_candidates[0][4]
            print(f"  • Best Raw Score (ΔG)       : {unique_candidates[0][0]:.4f} kcal/mol")
            if fe_atoms and has_het_n:
                fe_a = fe_atoms[0]
                het_details = []
                for a in best_pose:
                    if getattr(a, "is_het_n", False):
                        dist = math.sqrt((a.x - fe_a.x)**2 + (a.y - fe_a.y)**2 + (a.z - fe_a.z)**2)
                        un_fe = ((fe_a.x - a.x)/dist, (fe_a.y - a.y)/dist, (fe_a.z - a.z)/dist)
                        cos_theta = a.lp_dx * un_fe[0] + a.lp_dy * un_fe[1] + a.lp_dz * un_fe[2]
                        het_details.append((dist, cos_theta, getattr(a, "is_unhindered_n", False)))
                if het_details:
                    min_dist, best_cos, is_unh = min(het_details, key=lambda x: x[0])
                    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, best_cos))))
                    tag = "Unhindered" if is_unh else "Hindered"
                    print(f"  • Top Pose Fe-N Distance    : {min_dist:.3f} Å ({tag}, Vector Alignment: {angle_deg:.1f}° off Fe-axis)")
            if ref_ligand_atoms or ref_atoms:
                top_rmsd = calculate_heavy_atom_rmsd(unique_candidates[0][4], ref_ligand_atoms if ref_ligand_atoms else ref_atoms)
                print(f"  • Top Pose Ref RMSD         : {top_rmsd:.4f} Å")

    print(f"🎉 Total Execution Time   : {time.time() - t_start:.2f}s\n")
    return f"Processed: {lig_path.name} -> {target_workspace.name}"