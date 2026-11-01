#!/usr/bin/env python3
"""
pose_curator_cypware_tleap_amber.py

Full Protein-Ligand Complex Curation, Solvation, Neutralization & Amber Minimization Pipeline:
1. Passes docked ligands to Antechamber/Parmchk2 to generate GAFF2 parameters.
2. Builds Amber systems in tleap with full CYPWare split-heme coordination.
3. Solvates the system (OPC water box) and neutralizes with counter-ions.
4. Performs robust energy minimization using Amber's native engines (auto-detects pmemd.cuda, pmemd, or sander).
5. Parses output coordinates, reports added waters, ions, and energies.
6. Saves the minimized complex (aligned to reference) and safely computes RMSD via atom intersection mapping.
7. Validates poses via PoseBusters.
"""

import argparse
import glob
import os
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    rdDetermineBonds = None

try:
    import MDAnalysis as mda
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
except ImportError:
    mda = None

try:
    from posebusters import PoseBusters
except ImportError:
    PoseBusters = None

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'FE': 1.80
}
CLASH_TOLERANCE = 0.75


# =============================================================================
# STRUCTURE LOADING & TOPOLOGY ASSIGNMENT
# =============================================================================

def load_molecule_flexible(file_path, remove_hs=False, sanitize=True):
    ext = os.path.splitext(file_path)[1].lower()
    mol = None
    if ext == ".pdb":
        mol = Chem.MolFromPDBFile(file_path, removeHs=remove_hs, sanitize=sanitize)
        if mol is None and sanitize:
            mol = Chem.MolFromPDBFile(file_path, removeHs=remove_hs, sanitize=False)
    elif ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(file_path, removeHs=remove_hs, sanitize=sanitize)
        mol = next((m for m in suppl if m is not None), None) if suppl is not None else None
        if mol is None and sanitize:
            suppl = Chem.SDMolSupplier(file_path, removeHs=remove_hs, sanitize=False)
            mol = next((m for m in suppl if m is not None), None) if suppl is not None else None
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(file_path, removeHs=remove_hs, sanitize=sanitize)
        if mol is None and sanitize:
            mol = Chem.MolFromMol2File(file_path, removeHs=remove_hs, sanitize=False)
    return mol


def extract_base_identifier(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_ps\d+.*", "", stem)
    stem = re.sub(r"_fixed.*", "", stem)
    return stem


def find_matching_template_file(base_id: str, search_dirs: List[Path]) -> Optional[Path]:
    valid_exts = [".mol2", ".sdf", ".sd", ".pdb"]
    for s_dir in search_dirs:
        if not s_dir or not s_dir.is_dir():
            continue
        for ext in valid_exts:
            cand = s_dir / f"{base_id}{ext}"
            if cand.is_file():
                return cand
        for f in s_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                f_stem = f.stem.lower()
                target_stem = base_id.lower()
                if f_stem == f"{target_stem}_lig_aln" or f_stem.startswith(f"{target_stem}_") or f_stem == target_stem:
                    return f
    return None


def assign_bonds_and_kekulize(target_mol, template_mol):
    if template_mol is not None:
        target_heavy = Chem.RemoveHs(target_mol, sanitize=False)
        template_heavy = Chem.RemoveHs(template_mol, sanitize=False)
        if target_heavy.GetNumAtoms() == template_heavy.GetNumAtoms():
            t_elems = [a.GetSymbol() for a in target_heavy.GetAtoms()]
            ref_elems = [a.GetSymbol() for a in template_heavy.GetAtoms()]
            if t_elems == ref_elems:
                try:
                    assigned = AllChem.AssignBondOrdersFromTemplate(template_heavy, target_heavy)
                    clone = Chem.AddHs(assigned, addCoords=True)
                    Chem.SanitizeMol(clone)
                    return clone
                except Exception:
                    pass
        try:
            mcs = rdFMCS.FindMCS(
                [template_heavy, target_heavy],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                completeRingsOnly=False,
                timeout=5
            )
            query = Chem.MolFromSmarts(mcs.smartsString)
            ref_match = template_heavy.GetSubstructMatch(query)
            tgt_match = target_heavy.GetSubstructMatch(query)
            if len(ref_match) == template_heavy.GetNumAtoms() and len(tgt_match) == target_heavy.GetNumAtoms():
                clone = Chem.Mol(template_heavy)
                if clone.GetNumConformers() == 0:
                    conf = Chem.Conformer(clone.GetNumAtoms())
                    clone.AddConformer(conf, assignId=True)
                else:
                    conf = clone.GetConformer()
                t_conf = target_heavy.GetConformer()
                for r_idx, t_idx in zip(ref_match, tgt_match):
                    conf.SetAtomPosition(r_idx, t_conf.GetAtomPosition(t_idx))
                clone = Chem.AddHs(clone, addCoords=True)
                Chem.SanitizeMol(clone)
                return clone
        except Exception:
            pass

    if rdDetermineBonds is not None:
        try:
            work_mol = Chem.Mol(target_mol)
            if not any(a.GetAtomicNum() == 1 for a in work_mol.GetAtoms()):
                work_mol = Chem.AddHs(work_mol, addCoords=True)
            rdDetermineBonds.DetermineBondOrders(work_mol, charge=0)
            Chem.SanitizeMol(work_mol)
            return work_mol
        except Exception:
            pass
    try:
        Chem.SanitizeMol(target_mol)
    except Exception:
        pass
    return target_mol


# =============================================================================
# HEME / AXIAL CYS DETECTION
# =============================================================================

def extract_receptor_and_cofactor_atoms(protein_pdb_path: str):
    u = mda.Universe(protein_pdb_path)
    prot_atoms = u.select_atoms("protein or resname CM1 CYM CYP")
    cofactor_atoms = u.select_atoms("not (protein or resname CM1 CYM CYP) and not (resname HOH WAT TIP3 H2O)")
    combined_atoms = prot_atoms + cofactor_atoms
    env_coords = combined_atoms.positions
    env_vdw = np.array([
        VDW_RADII.get(a.element.strip().upper() if hasattr(a, 'element') and a.element else a.name[0].upper(), 1.70)
        for a in combined_atoms
    ])
    print(f"--> [RECEPTOR] Parsed {len(prot_atoms)} protein atoms and {len(cofactor_atoms)} cofactor/metal atoms.")
    return env_coords, env_vdw


def parse_pdb_residues_and_find_axial_cys(pdb_path, fe_cutoff=3.0):
    atoms = []
    residues = []
    seen = set()
    for line in Path(pdb_path).read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom = line[12:16].strip()
        resn = line[17:20].strip().upper()
        chain = line[21:22]
        resid = line[22:26].strip()
        icode = line[26:27]
        key = (chain, resid, icode, resn)
        if key not in seen:
            seen.add(key)
            residues.append(key)
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        atoms.append({"atom": atom, "resn": resn, "key": key, "xyz": xyz, "line": line})

    ordinal = {key: i + 1 for i, key in enumerate(residues)}

    fe_candidates = [
        a for a in atoms
        if a["atom"].upper() in {"FE", "FE1"}
        and a["resn"] in {"FE1", "FE", "HM1", "HEM", "HEA", "HEB", "HEC", "HEO"}
    ]
    if len(fe_candidates) != 1:
        raise RuntimeError(f"Expected exactly one heme Fe atom; found {len(fe_candidates)}.")
    fe = fe_candidates[0]

    heme_ring_candidates = []
    for key in residues:
        if key[3] in {"HM1", "HEM", "HEA", "HEB", "HEC", "HEO"}:
            res_atoms = {a["atom"].upper() for a in atoms if a["key"] == key}
            if {"NA", "NB", "NC", "ND"}.issubset(res_atoms):
                heme_ring_candidates.append(key)

    if len(heme_ring_candidates) != 1:
        raise RuntimeError(f"Expected 1 porphyrin residue with NA/NB/NC/ND; found {len(heme_ring_candidates)}.")
    heme_key = heme_ring_candidates[0]

    cys_sg = [
        a for a in atoms
        if a["resn"] in {"CM1", "CYS", "CYM", "CYP"} and a["atom"].upper() == "SG"
    ]
    if not cys_sg:
        raise RuntimeError("No CM1/CYS/CYM/CYP SG atom found in receptor.")
    distances = [(float(np.linalg.norm(a["xyz"] - fe["xyz"])), a) for a in cys_sg]
    dist, cys = min(distances, key=lambda x: x[0])
    if dist > fe_cutoff:
        raise RuntimeError(f"Nearest Cys SG is {dist:.2f} Å from heme Fe (cutoff {fe_cutoff:.2f} Å)")

    is_split_fe = (fe["key"] != heme_key)

    return {
        "heme_residue_index": ordinal[heme_key],
        "heme_residue_key": heme_key,
        "fe_residue_index": ordinal[fe["key"]],
        "fe_residue_key": fe["key"],
        "fe_atom_name": fe["atom"].upper(),
        "is_split_fe": is_split_fe,
        "cys_residue_index": ordinal[cys["key"]],
        "cys_residue_key": cys["key"],
        "fe_s_distance_A": dist,
        "heme_n_atom_names": ["NA", "NB", "NC", "ND"],
    }


def make_leap_compatible_receptor(source_pdb, output_pdb, axial_info, has_cm1_hm1_fe1=True):
    cys_key = axial_info["cys_residue_key"]
    heme_key = axial_info["heme_residue_key"]
    fe_key = axial_info["fe_residue_key"]
    is_split_fe = axial_info["is_split_fe"]

    lines = []
    prev_key = None

    for line in Path(source_pdb).read_text(errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atom = line[12:16].strip().upper()
            resn = line[17:20].strip().upper()
            chain = line[21:22]
            resid = line[22:26].strip()
            icode = line[26:27]
            key = (chain, resid, icode, resn)

            if prev_key is not None and key != prev_key:
                if key == heme_key or (is_split_fe and key == fe_key):
                    if not (lines and lines[-1].startswith("TER")):
                        lines.append("TER")

            if key == cys_key:
                if atom in {"HG", "HG1"}:
                    continue
                target_cys = "CM1" if has_cm1_hm1_fe1 else "CYM"
                line = line[:17] + target_cys.ljust(3) + line[20:]

            elif key == heme_key:
                target_heme = "HM1" if has_cm1_hm1_fe1 else "HEM"
                line = line[:17] + target_heme.ljust(3) + line[20:]

            elif is_split_fe and key == fe_key:
                try:
                    new_resid = f"{int(resid) + 1:4d}"
                except ValueError:
                    new_resid = " 999"
                line = line[:17] + "FE1" + line[20:22] + new_resid + line[26:]

            lines.append(line)
            prev_key = key

        elif line.startswith("TER"):
            if not (lines and lines[-1].startswith("TER")):
                lines.append("TER")

    if not (lines and lines[-1].startswith("TER")):
        lines.append("TER")
    lines.append("END")
    Path(output_pdb).write_text("\n".join(lines) + "\n")


def discover_amber_heme_files(param_dir):
    p = Path(param_dir)
    files = [x for x in p.iterdir() if x.is_file()]
    mol2_map = {}
    for x in files:
        if x.suffix.lower() == ".mol2":
            mol2_map[x.stem.upper()] = x
    frcmods = sorted(x for x in files if x.suffix.lower() == ".frcmod")
    libs = sorted(x for x in files if x.suffix.lower() in {".lib", ".off"})
    preps = sorted(x for x in files if x.suffix.lower() in {".prepi", ".prep", ".in"})
    return mol2_map, frcmods, libs, preps


# =============================================================================
# AMBERTOOLS & SANDER MINIMIZATION & SOLVATION
# =============================================================================

def get_available_amber_binary():
    """Automatically selects the fastest available Amber minimization binary."""
    if shutil.which("pmemd.cuda"):
        return "pmemd.cuda"
    elif shutil.which("pmemd"):
        return "pmemd"
    elif shutil.which("sander"):
        return "sander"
    else:
        raise RuntimeError("No Amber minimization binary (pmemd.cuda, pmemd, sander) found in PATH.")


def write_exact_ps003_pdb(rdkit_mol, out_path):
    """Writes out the PDB file matching the exact column placement of ps003.pdb."""
    lines = []
    for idx, atom in enumerate(rdkit_mol.GetAtoms(), start=1):
        pos = rdkit_mol.GetConformer().GetAtomPosition(idx - 1)
        name = atom.GetProp("_Name") if atom.HasProp("_Name") else f"{atom.GetSymbol().upper()}{idx}"
        name_f = f"{name:<4s}"
        elem = atom.GetSymbol().upper()
        line = f"HETATM{idx:5d} {name_f} UNL    1       {pos.x:7.3f} {pos.y:7.3f}  {pos.z:6.3f}  1.00  0.00           {elem:1s} \n"
        lines.append(line)
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))


def run_tleap_build(receptor_pdb, sanitized_rdkit_mol, workdir, axial_info,
                    param_dir, ligand_charge=0, antechamber_charge="gas",
                    solvate_buffer=10.0, water_model="OPCBOX"):
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    mol2_map, frcmod_paths, libs, preps = discover_amber_heme_files(param_dir)

    lig_h = Chem.AddHs(Chem.Mol(sanitized_rdkit_mol), addCoords=True)
    try:
        Chem.SanitizeMol(lig_h)
    except Exception:
        pass

    calc_charge = Chem.GetFormalCharge(lig_h)
    if int(ligand_charge) != calc_charge:
        ligand_charge = calc_charge

    for i, atom in enumerate(lig_h.GetAtoms()):
        res_info = Chem.AtomPDBResidueInfo()
        res_info.SetResidueName("UNL")
        res_info.SetResidueNumber(1)
        name = f"{atom.GetSymbol().upper()[:2]}{i+1}".ljust(4)
        res_info.SetName(name)
        res_info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(res_info)
        atom.SetProp("_Name", name.strip())

    ligand_sdf_in = work / "ligand_in.sdf"
    w = Chem.SDWriter(str(ligand_sdf_in))
    w.write(lig_h)
    w.close()

    ligand_pdb_in = work / "ligand_in.pdb"
    Chem.MolToPDBFile(lig_h, str(ligand_pdb_in))

    lig_mol2_out = work / "LIG_GAFF2.mol2"
    lig_frcmod = work / "LIG_GAFF2.frcmod"
    ac_log = work / "antechamber.log"
    ac_log.write_text("")

    def run_ac(charge_model, fmt, infile_name):
        cmd = [
            "antechamber", "-i", infile_name, "-fi", fmt,
            "-o", lig_mol2_out.name, "-fo", "mol2",
            "-c", charge_model, "-nc", str(int(ligand_charge)),
            "-rn", "UNL", "-at", "gaff2"
        ]
        res = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True)
        with open(ac_log, "a") as f:
            f.write(f"\n--- CMD: {' '.join(cmd)} ---\n")
            f.write(res.stdout + "\n" + res.stderr + "\n")
        return res

    res = run_ac(antechamber_charge, "sdf", ligand_sdf_in.name)
    if res.returncode != 0 and antechamber_charge == "bcc":
        res = run_ac("gas", "sdf", ligand_sdf_in.name)
    if res.returncode != 0:
        res = run_ac(antechamber_charge, "pdb", ligand_pdb_in.name)
    if res.returncode != 0 and antechamber_charge == "bcc":
        res = run_ac("gas", "pdb", ligand_pdb_in.name)
    if res.returncode != 0:
        raise RuntimeError(f"Antechamber failed. Inspect log: {ac_log}")

    res_parmchk = subprocess.run(
        ["parmchk2", "-i", lig_mol2_out.name, "-f", "mol2", "-o", lig_frcmod.name, "-s", "gaff2"],
        cwd=str(work), capture_output=True, text=True
    )
    (work / "parmchk2.log").write_text(res_parmchk.stdout + "\n" + res_parmchk.stderr)
    if res_parmchk.returncode != 0:
        raise RuntimeError(f"Parmchk2 failed. Inspect log: {work}/parmchk2.log")

    has_cm1_hm1_fe1 = all(k in mol2_map for k in ("CM1", "HM1", "FE1"))
    receptor_leap = work / "receptor_leap.pdb"
    make_leap_compatible_receptor(receptor_pdb, receptor_leap, axial_info, has_cm1_hm1_fe1=has_cm1_hm1_fe1)

    leap_lines = [
        "source leaprc.protein.ff19SB",
        "source leaprc.gaff2",
        "source leaprc.water.opc",
    ]
    for name_key, mol2_path in sorted(mol2_map.items()):
        leap_lines.append(f'{name_key} = loadmol2 "{mol2_path.resolve()}"')
    for lib_path in libs:
        leap_lines.append(f'loadoff "{lib_path.resolve()}"')
    for prep_path in preps:
        leap_lines.append(f'loadamberprep "{prep_path.resolve()}"')
    for frc in frcmod_paths:
        leap_lines.append(f'loadamberparams "{frc.resolve()}"')

    leap_lines += [
        f'loadamberparams "{lig_frcmod.name}"',
        f'LIG = loadmol2 "{lig_mol2_out.name}"',
        f'HEM_SYS = loadpdb "{receptor_leap.name}"',
        "COMPLEX = combine { HEM_SYS LIG }",
    ]

    hidx = axial_info["heme_residue_index"]
    cidx = axial_info["cys_residue_index"]
    fidx = axial_info["fe_residue_index"]
    fe_name = axial_info["fe_atom_name"]

    if axial_info["is_split_fe"]:
        leap_lines.append(f"bond COMPLEX.{cidx-1}.C COMPLEX.{cidx}.N")
        leap_lines.append(f"bond COMPLEX.{cidx}.C COMPLEX.{cidx+1}.N")
        for n in ("NA", "NB", "NC", "ND"):
            leap_lines.append(f"bond COMPLEX.{hidx}.{n} COMPLEX.{fidx}.{fe_name}")
        leap_lines.append(f"bond COMPLEX.{cidx}.SG COMPLEX.{fidx}.{fe_name}")
    else:
        for n in ("NA", "NB", "NC", "ND"):
            leap_lines.append(f"bond COMPLEX.{hidx}.{fe_name} COMPLEX.{hidx}.{n}")
        leap_lines.append(f"bond COMPLEX.{cidx}.SG COMPLEX.{hidx}.{fe_name}")

    # Add Solvation and Neutralization
    leap_lines += [
        f"solvatebox COMPLEX {water_model} {solvate_buffer}",
        "addions COMPLEX Na+ 0",
        "addions COMPLEX Cl- 0",
        "check COMPLEX",
        'saveamberparm COMPLEX "complex.prmtop" "complex.inpcrd"',
        'savepdb COMPLEX "complex_tleap.pdb"',
        "quit",
    ]
    leap_in = work / "tleap.in"
    leap_in.write_text("\n".join(leap_lines) + "\n")

    result = subprocess.run(["tleap", "-f", leap_in.name], cwd=str(work), capture_output=True, text=True)
    (work / "leap.log").write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0 or not (work / "complex.prmtop").exists():
        raise RuntimeError(f"tleap failed. Inspect log: {work}/leap.log")

    return work / "complex.prmtop", work / "complex.inpcrd", lig_h


def extract_reference_ligand_from_complex(ref_complex_pdb: str, workdir: Path) -> Optional[Chem.Mol]:
    """Extracts the ligand molecule from the reference complex directly to a physical file to avoid MDAnalysis IO errors."""
    if not mda: return None
    u_ref = mda.Universe(ref_complex_pdb)
    lig_ag = u_ref.select_atoms("not (protein or nucleic or resname HOH WAT TIP3 H2O OPC NA CL K MG ZN CA SO4 PO4 HEM HM1 FE1)")
    if len(lig_ag) == 0:
        return None
    
    temp_pdb = workdir / "temp_ref_lig.pdb"
    lig_ag.write(str(temp_pdb))
    mol = Chem.MolFromPDBFile(str(temp_pdb))
    if temp_pdb.exists():
        temp_pdb.unlink()
    return mol


def amber_sander_minimize(protein_pdb, rdkit_mol, out_complex_pdb, out_fixed_pdb,
                          max_iterations=2000, param_dir=None, ligand_charge=0,
                          axial_info=None, workdir=None, antechamber_charge="gas",
                          solvate_buffer=10.0, water_model="OPCBOX", ref_complex_pdb=None,
                          target_template_mol=None):
    
    work = Path(workdir or Path(out_complex_pdb).with_suffix("")).resolve()
    prmtop, inpcrd, lig_h = run_tleap_build(
        protein_pdb, rdkit_mol, work, axial_info, param_dir,
        ligand_charge=ligand_charge, antechamber_charge=antechamber_charge,
        solvate_buffer=solvate_buffer, water_model=water_model
    )

    num_waters, num_ions = 0, 0
    if mda is not None:
        try:
            u_built = mda.Universe(str(work / "complex_tleap.pdb"))
            num_waters = len(u_built.select_atoms("resname HOH WAT TIP3 H2O OPC"))
            num_ions = len(u_built.select_atoms("resname Na+ Cl- NA CL K+ K MG2+ MG ZN2+ ZN"))
        except Exception:
            pass

    min_in_path = work / "min.in"
    min_in_content = f"""Initial minimization of solvated complex
 &cntrl
  imin=1, maxcyc={max_iterations}, ncyc=500,
  ntmin=2, cut=10.0,
  ntb=1, ntwx=50,
 /
"""
    min_in_path.write_text(min_in_content)

    print(f"\n      --> [Amber Minimization Protocol]")
    print(f"          Water Molecules Added        : {num_waters}")
    print(f"          Counter Ions Added           : {num_ions}")

    amber_bin = get_available_amber_binary()
    print(f"      --> [Amber Minimization Engine]: Using '{amber_bin}'")

    sander_cmd = [
        amber_bin, "-O",
        "-i", "min.in",
        "-o", "min.out",
        "-p", "complex.prmtop",
        "-c", "complex.inpcrd",
        "-r", "min.rst7",
        "-x", "min.nc"
    ]
    
    res = subprocess.run(sander_cmd, cwd=str(work), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Minimization failed. Check {work}/min.out")

    e_init, e_final = 0.0, 0.0
    min_out_text = (work / "min.out").read_text(errors="replace")
    energies = re.findall(r"NSTEP\s+ENERGY\s+RMS\s+GMAX[^\n]+\n\s+([-\d\.]+)\s+([-\d\.]+)", min_out_text)
    if energies:
        e_init = float(energies[0][1])
        e_final = float(energies[-1][1])

    print(f"          Pre-Minimization Potential E : {e_init:14.3f} kcal/mol")
    print(f"          Post-Minimization Potential E: {e_final:14.3f} kcal/mol")

    min_pdb_path = work / "minimized_complex.pdb"
    try:
        import parmed as pmd
        parm = pmd.load_file(str(work / "complex.prmtop"), str(work / "min.rst7"))
        parm.save(str(min_pdb_path), overwrite=True)
    except Exception:
        subprocess.run(f"ambpdb -p complex.prmtop -c min.rst7 > minimized_complex.pdb", shell=True, cwd=str(work))

    u_min = mda.Universe(str(min_pdb_path))
    u_lig = u_min.select_atoms("resname UNL")
    
    heavy_idx = [a.GetIdx() for a in lig_h.GetAtoms() if a.GetAtomicNum() > 1]
    if len(u_lig.positions) == lig_h.GetNumAtoms():
        c_before = np.asarray(lig_h.GetConformer().GetPositions())[heavy_idx]
        c_after_unaligned = u_lig.positions[heavy_idx]
        shift_rmsd = float(np.sqrt(np.mean(np.sum((c_after_unaligned - c_before) ** 2, axis=1))))
    else:
        shift_rmsd = float('nan')

    # 2. Origin-Centered Translation and Alignment Math
    rmsd_metrics = {}
    if ref_complex_pdb and Path(ref_complex_pdb).is_file() and mda is not None:
        try:
            u_ref = mda.Universe(str(ref_complex_pdb))
            
            # Align by shared Backbone atoms
            bb_min = u_min.select_atoms("protein and backbone and not name H*")
            bb_ref = u_ref.select_atoms("protein and backbone and not name H*")
            
            dict_min = {(a.resnum, a.name): a.index for a in bb_min}
            dict_ref = {(a.resnum, a.name): a.index for a in bb_ref}
            common_bb = set(dict_min.keys()).intersection(set(dict_ref.keys()))
            
            if common_bb:
                from MDAnalysis.analysis.align import rotation_matrix
                idx_min = [dict_min[k] for k in sorted(common_bb)]
                idx_ref = [dict_ref[k] for k in sorted(common_bb)]
                
                pos_min = u_min.atoms[idx_min].positions
                pos_ref = u_ref.atoms[idx_ref].positions
                
                # Center both selections to origin BEFORE generating rotation matrix
                com_min = pos_min.mean(axis=0)
                com_ref = pos_ref.mean(axis=0)
                
                pos_min_centered = pos_min - com_min
                pos_ref_centered = pos_ref - com_ref
                
                R, _ = rotation_matrix(pos_min_centered, pos_ref_centered)
                
                # Apply transformation sequence to the entire u_min universe
                u_min.atoms.translate(-com_min)
                u_min.atoms.rotate(R)
                u_min.atoms.translate(com_ref)

            def calc_intersect_rmsd(sel_str):
                ag_min = u_min.select_atoms(sel_str)
                ag_ref = u_ref.select_atoms(sel_str)
                d_min = {(a.resnum, a.name): a.index for a in ag_min}
                d_ref = {(a.resnum, a.name): a.index for a in ag_ref}
                common = set(d_min.keys()).intersection(set(d_ref.keys()))
                if not common: return float("nan")
                i_min = [d_min[k] for k in sorted(common)]
                i_ref = [d_ref[k] for k in sorted(common)]
                return float(mda.analysis.rms.rmsd(u_min.atoms[i_min].positions, u_ref.atoms[i_ref].positions, superposition=False))

            rmsd_metrics["all_atom"] = calc_intersect_rmsd("protein and not name H*")
            rmsd_metrics["backbone"] = calc_intersect_rmsd("protein and backbone and not name H*")
            rmsd_metrics["sidechain"] = calc_intersect_rmsd("protein and not backbone and not name H*")
            
            u_min.atoms.write(out_complex_pdb)
            
        except Exception as exc:
            print(f"      [WARNING] RMSD comparison failed: {exc}")
            shutil.copy(min_pdb_path, out_complex_pdb)
    else:
        shutil.copy(min_pdb_path, out_complex_pdb)

    # 3. Extract Ligand Coordinates from Aligned Universe
    lig_out = Chem.Mol(lig_h)
    conf = lig_out.GetConformer()
    for i, atom in enumerate(lig_out.GetAtoms()):
        if i < len(u_lig):
            pos = u_lig[i].position
            conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))

    write_exact_ps003_pdb(lig_out, out_fixed_pdb)

    # 4. Safely Compare Ligand RMSD to the extracted Reference Crystal Ligand
    if ref_complex_pdb:
        ref_lig_mol = target_template_mol
        if ref_lig_mol is None:
            ref_lig_mol = extract_reference_ligand_from_complex(str(ref_complex_pdb), work)
            
        if ref_lig_mol is not None:
            try:
                m1 = Chem.RemoveHs(ref_lig_mol)
                m2 = Chem.RemoveHs(lig_out)
                mcs = rdFMCS.FindMCS([m1, m2], atomCompare=rdFMCS.AtomCompare.CompareElements)
                query = Chem.MolFromSmarts(mcs.smartsString)
                match1 = m1.GetSubstructMatch(query)
                match2 = m2.GetSubstructMatch(query)
                if match1 and match2:
                    c1 = m1.GetConformer().GetPositions()[list(match1)]
                    c2 = m2.GetConformer().GetPositions()[list(match2)]
                    rmsd_metrics["ligand"] = float(np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=1))))
                else:
                    rmsd_metrics["ligand"] = float('nan')
            except Exception:
                rmsd_metrics["ligand"] = float('nan')
        else:
            rmsd_metrics["ligand"] = float('nan')

    if ref_complex_pdb:
        print(f"      --> [Reference RMSD Comparison]")
        print(f"          Protein Atom RMSD     : {rmsd_metrics.get('all_atom', float('nan')):.3f} Å")
        print(f"          Protein Backbone RMSD : {rmsd_metrics.get('backbone', float('nan')):.3f} Å")
        print(f"          Protein Sidechain RMSD: {rmsd_metrics.get('sidechain', float('nan')):.3f} Å")
        print(f"          Ligand RMSD           : {rmsd_metrics.get('ligand', float('nan')):.3f} Å")

    return Chem.RemoveHs(lig_out), shift_rmsd, {"waters": num_waters, "ions": num_ions, "e_init": e_init, "e_final": e_final, **rmsd_metrics}


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Complex Solvation, Neutralization & Amber Minimization Pipeline.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing docked PDB/SDF files")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save fixed files and report")
    parser.add_argument("-p", "--protein", required=True, help="Receptor protein PDB file")
    parser.add_argument("-r", "--ref", default=None, help="Reference crystal ligand file or directory")
    parser.add_argument("--ref-complex", default=None, help="Reference full complex PDB for RMSD comparison")
    parser.add_argument("-l", "--ligands-dir", default=None, help="Directory containing source mol2/sdf files")
    parser.add_argument("--smiles", default=None, help="Optional canonical SMILES string for template")
    parser.add_argument("--pattern", default=None, help="Glob pattern, e.g. *.pdb")
    parser.add_argument("--skip-bust", action="store_true", help="Skip PoseBusters verification")
    parser.add_argument("--no-relax", action="store_true", help="Skip Amber minimization")
    parser.add_argument("--max-min-iterations", type=int, default=2000)
    parser.add_argument("--heme-params-dir", default=None)
    parser.add_argument("--ligand-charge", type=int, default=0)
    parser.add_argument("--antechamber-charge", default="gas", choices=["bcc", "gas", "rc"])
    parser.add_argument("--fe-s-cutoff", type=float, default=3.0)
    parser.add_argument("--solvate-buffer", type=float, default=10.0, help="Water box buffer distance in Angstroms")
    parser.add_argument("--water-model", default="OPCBOX", help="Water model (e.g. OPCBOX)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"[ERROR] Input directory does not exist: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if mda is None:
        sys.exit("[ERROR] MDAnalysis is required.")

    print("=" * 80)
    print("🚀 INITIATING FULL COMPLEX SOLVATION, NEUTRALIZATION & AMBER MINIMIZATION")
    print("=" * 80)

    if not os.path.isfile(args.protein):
        sys.exit(f"[ERROR] Protein file not found: {args.protein}")
    env_coords, env_vdw = extract_receptor_and_cofactor_atoms(args.protein)

    if not args.no_relax:
        if not args.heme_params_dir or not Path(args.heme_params_dir).is_dir():
            sys.exit(f"[ERROR] Valid --heme-params-dir is required: {args.heme_params_dir}")
        try:
            axial_info = parse_pdb_residues_and_find_axial_cys(args.protein, fe_cutoff=args.fe_s_cutoff)
            print("[HEME] Detected Fe/Cys coordination:")
            print(f"       Porphyrin ordinal  : {axial_info['heme_residue_index']} ({axial_info['heme_residue_key'][3]})")
            print(f"       Fe residue ordinal : {axial_info['fe_residue_index']} ({axial_info['fe_residue_key'][3]}, split={axial_info['is_split_fe']})")
            print(f"       Axial Cys ordinal  : {axial_info['cys_residue_index']} ({axial_info['cys_residue_key'][3]})")
            print(f"       Fe-S distance      : {axial_info['fe_s_distance_A']:.2f} Å")
        except Exception as exc:
            sys.exit(f"[ERROR] Heme/axial-Cys detection failed: {exc}")
    else:
        axial_info = None

    patterns = [args.pattern] if args.pattern else ["*.pdb", "*.sdf", "*.sd", "*.PDB", "*.SDF", "*.SD"]
    pose_files = sorted(set(f for pat in patterns for f in glob.glob(os.path.join(args.input_dir, pat))))
    pose_files = [f for f in pose_files if not Path(f).stem.endswith("_fixed")
                  and os.path.realpath(f) != os.path.realpath(args.protein)
                  and (not args.ref or os.path.realpath(f) != os.path.realpath(args.ref))]

    template_dirs = []
    if args.ligands_dir and os.path.isdir(args.ligands_dir):
        template_dirs.append(Path(args.ligands_dir))
    if args.ref and os.path.isdir(args.ref):
        template_dirs.append(Path(args.ref))
    for common_path in [Path("../mol2_files"), Path("../../mol2_files"), Path("../extracted_reference_ligands1W0E")]:
        if common_path.is_dir():
            template_dirs.append(common_path)

    single_ref_mol = None
    if args.ref and os.path.isfile(args.ref):
        single_ref_mol = load_molecule_flexible(args.ref, remove_hs=False, sanitize=False)

    buster = PoseBusters(config="dock") if (PoseBusters and not args.skip_bust) else None
    summary_records = []

    print("\n" + "=" * 80)
    print(f"{'POSE FILENAME':<32} | {'WATERS':<8} | {'IONS':<6} | {'E_FINAL (kcal/mol)':<18} | {'STATUS':<10}")
    print("-" * 80)

    for pose_path in pose_files:
        base_name = os.path.basename(pose_path)
        out_name_pdb = os.path.splitext(base_name)[0] + "_fixed.pdb"
        out_name_sdf = os.path.splitext(base_name)[0] + "_fixed.sdf"
        out_path_pdb = os.path.join(args.output_dir, out_name_pdb)
        out_path_sdf = os.path.join(args.output_dir, out_name_sdf)
        complex_path = os.path.join(args.output_dir, os.path.splitext(base_name)[0] + "_complex_solvated.pdb")
        workdir = Path(args.output_dir) / (Path(base_name).stem + "_amber_build")

        mol = load_molecule_flexible(pose_path, remove_hs=False, sanitize=False)
        if mol is None or mol.GetNumConformers() == 0:
            print(f"{base_name:<32} | READ ERR | Could not parse molecule")
            continue

        base_id = extract_base_identifier(base_name)
        template_file = find_matching_template_file(base_id, template_dirs)
        target_template_mol = None

        if template_file:
            target_template_mol = load_molecule_flexible(str(template_file), remove_hs=False, sanitize=False)
        elif single_ref_mol is not None:
            target_template_mol = single_ref_mol
        elif args.smiles:
            try:
                target_template_mol = Chem.MolFromSmiles(args.smiles)
            except Exception:
                pass

        mol = assign_bonds_and_kekulize(mol, target_template_mol)
        record = {"Filename": base_name, "Fixed_PDB_Path": out_path_pdb, "Fixed_SDF_Path": out_path_sdf}

        if args.no_relax:
            fixed_mol = mol
            write_exact_ps003_pdb(fixed_mol, out_path_pdb)
            w = Chem.SDWriter(out_path_sdf)
            w.write(fixed_mol)
            w.close()
            record["Waters_Added"] = 0
            record["Ions_Added"] = 0
            record["Final_Energy_kcal_mol"] = 0.0
        else:
            try:
                fixed_mol, shift_rmsd, metrics = amber_sander_minimize(
                    args.protein, mol, complex_path, out_path_pdb,
                    max_iterations=args.max_min_iterations,
                    param_dir=args.heme_params_dir,
                    ligand_charge=args.ligand_charge,
                    axial_info=axial_info,
                    workdir=workdir,
                    antechamber_charge=args.antechamber_charge,
                    solvate_buffer=args.solvate_buffer,
                    water_model=args.water_model,
                    ref_complex_pdb=args.ref_complex,
                    target_template_mol=target_template_mol
                )

                w = Chem.SDWriter(out_path_sdf)
                w.write(fixed_mol)
                w.close()

                record["Complex_Solvated_PDB"] = complex_path
                record["Waters_Added"] = metrics.get("waters", 0)
                record["Ions_Added"] = metrics.get("ions", 0)
                record["Energy_Init_kcal_mol"] = metrics.get("e_init", 0.0)
                record["Energy_Final_kcal_mol"] = metrics.get("e_final", 0.0)
                record["RMSD_AllAtom"] = metrics.get("all_atom", float("nan"))
                record["RMSD_Backbone"] = metrics.get("backbone", float("nan"))
                record["RMSD_Sidechain"] = metrics.get("sidechain", float("nan"))
                record["RMSD_Ligand"] = metrics.get("ligand", float("nan"))

                print(f"{base_name:<32} | {metrics.get('waters', 0):<8} | {metrics.get('ions', 0):<6} | {metrics.get('e_final', 0.0):<18.2f} | {'MINIMIZED':<10}")

                if workdir.exists():
                    shutil.rmtree(workdir)

            except Exception as e:
                err_msg = str(e).splitlines()[0][:55]
                print(f"{base_name:<32} | ERROR    | {err_msg}")
                record["Status"] = "ERROR"
                summary_records.append(record)
                continue

        summary_records.append(record)

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        csv_out = os.path.join(args.output_dir, "batch_solvation_minimization_summary.csv")
        df_summary.to_csv(csv_out, index=False)
        print("=" * 80)
        print(f"\n[DONE] Batch solvation & Amber minimization complete across {len(summary_records)} poses.")
        print(f"• Solvated complexes saved to : {args.output_dir}/")
        print(f"• Summary Table saved to      : {csv_out}\n")


if __name__ == "__main__":
    main()