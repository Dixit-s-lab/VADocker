#!/usr/bin/env python3
"""
pose_curator_cypware_tleap_openmm3.py

Batch processor and PoseBusters validator for docked poses:
1. Passes docked ligands to Antechamber/Parmchk2 to generate GAFF2 parameters.
2. Builds Amber systems in tleap with full CYPWare split-heme (CM1 + HM1 + FE1) coordination.
3. Minimizes ONLY ligand atoms in OpenMM (Mass = 0.0 Da for protein, cofactors, HM1, FE1, CM1).
4. Maps coordinates back by strict topological index (eliminating scrambled atom names).
5. Exports both a sanitized, bond-order-valid SDF (for PoseBusters) and an exact ps003-format PDB.
"""

import argparse
import glob
import os
import re
import sys
import shutil
import subprocess
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
# AMBERTOOLS & OPENMM MINIMIZATION
# =============================================================================

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
                    param_dir, ligand_charge=0, antechamber_charge="gas"):
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
        "source leaprc.protein.ff14SB",
        "source leaprc.gaff2",
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

    leap_lines += [
        "check COMPLEX",
        'saveamberparm COMPLEX "complex.prmtop" "complex.inpcrd"',
        'savepdb COMPLEX "complex_tleap.pdb"',
        "quit",
    ]
    leap_in = work / "tleap.in"
    leap_in.write_text("\n".join(leap_lines) + "\n")

    result = subprocess.run(["tleap", "-f", leap_in.name], cwd=str(work), capture_output=True, text=True)
    (work / "tleap.log").write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0 or not (work / "complex.prmtop").exists():
        raise RuntimeError(f"tleap failed. Inspect log: {work}/tleap.log")

    return work / "complex.prmtop", work / "complex.inpcrd", lig_h


def openmm_ligand_only_minimize(protein_pdb, rdkit_mol, out_complex_pdb, out_fixed_pdb,
                                max_iterations=2000, restraint_k=0.0,
                                platform_name="CPU", param_dir=None,
                                ligand_charge=0, axial_info=None,
                                workdir=None, antechamber_charge="gas"):
    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        raise RuntimeError("Install OpenMM to perform the Amber-topology minimization.") from exc

    work = Path(workdir or Path(out_complex_pdb).with_suffix("")).resolve()
    prmtop, inpcrd, lig_h = run_tleap_build(
        protein_pdb, rdkit_mol, work, axial_info, param_dir,
        ligand_charge=ligand_charge, antechamber_charge=antechamber_charge
    )

    parm = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(inpcrd))
    system = parm.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    topology = parm.topology

    ligand_residues = [r for r in topology.residues() if r.name == "UNL"]
    if len(ligand_residues) != 1:
        raise RuntimeError(f"Expected one UNL residue in Amber topology; found {len(ligand_residues)}")

    topo_lig_atoms = list(ligand_residues[0].atoms())
    ligand_indices = {a.index for a in topo_lig_atoms}
    total_particles = system.getNumParticles()

    # FREEZE ENTIRE RECEPTOR: Protein, HM1, FE1, CM1 all have mass = 0.0
    frozen_count = 0
    for idx in range(total_particles):
        if idx not in ligand_indices:
            system.setParticleMass(idx, 0.0 * unit.dalton)
            frozen_count += 1

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except Exception:
        platform = openmm.Platform.getPlatformByName("CPU")

    context = openmm.Context(system, integrator, platform)
    context.setPositions(crd.positions)

    # Report pre-minimization system energy
    state_init = context.getState(getEnergy=True)
    e_init = state_init.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"\n      --> [OpenMM Minimization Setup]")
    print(f"          Total System Particles       : {total_particles}")
    print(f"          Frozen Receptor Particles    : {frozen_count} (Mass = 0.0 Da)")
    print(f"          Mobile Ligand Particles (UNL): {len(ligand_indices)} (GAFF2 Mobile)")
    print(f"          Pre-Minimization Potential E : {e_init:14.3f} kJ/mol")

    openmm.LocalEnergyMinimizer.minimize(context, tolerance=5.0, maxIterations=int(max_iterations))

    # Report post-minimization system energy
    state_final = context.getState(getPositions=True, getEnergy=True)
    e_final = state_final.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    delta_e = e_final - e_init
    print(f"          Post-Minimization Potential E: {e_final:14.3f} kJ/mol (ΔE = {delta_e:14.3f} kJ/mol)")

    positions = state_final.getPositions(asNumpy=True)

    with open(out_complex_pdb, "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)

    coords_A = positions.value_in_unit(unit.angstrom)

    # Topological 1-to-1 extraction (preserves exact chemical geometry and avoids atom-name scrambling)
    lig_out = Chem.Mol(lig_h)
    conf = lig_out.GetConformer()
    for i, topo_atom in enumerate(topo_lig_atoms):
        pos = coords_A[topo_atom.index]
        conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))

    # Output exact ps003-compatible PDB
    write_exact_ps003_pdb(lig_out, out_fixed_pdb)

    heavy = [a.GetIdx() for a in lig_out.GetAtoms() if a.GetAtomicNum() > 1]
    before = np.asarray(lig_h.GetConformer().GetPositions())
    after = np.asarray(lig_out.GetConformer().GetPositions())
    rmsd = float(np.sqrt(np.mean(np.sum((after[heavy] - before[heavy]) ** 2, axis=1))))

    return Chem.RemoveHs(lig_out), rmsd


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch curate docked poses and minimize ligand in OpenMM.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing docked PDB/SDF files")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save fixed .pdb/.sdf files and CSV report")
    parser.add_argument("-p", "--protein", required=True, help="Receptor protein PDB file (e.g. 3NXU_min.pdb)")
    parser.add_argument("-r", "--ref", default=None, help="Reference crystal ligand SDF/PDB or directory of references")
    parser.add_argument("-l", "--ligands-dir", default=None, help="Directory containing source mol2/sdf files")
    parser.add_argument("--smiles", default=None, help="Optional canonical SMILES string for template")
    parser.add_argument("--pattern", default=None, help="Glob pattern, e.g. *.pdb; default: all .pdb/.sdf/.sd")
    parser.add_argument("--skip-bust", action="store_true", help="Skip running PoseBusters verification")
    parser.add_argument("--no-relax", action="store_true", help="Skip OpenMM ligand minimization")
    parser.add_argument("--openmm-platform", default="CPU", choices=["CPU", "CUDA", "OpenCL"])
    parser.add_argument("--max-min-iterations", type=int, default=2000)
    parser.add_argument("--heme-params-dir", default=None)
    parser.add_argument("--ligand-charge", type=int, default=0)
    parser.add_argument("--antechamber-charge", default="gas", choices=["bcc", "gas", "rc"])
    parser.add_argument("--fe-s-cutoff", type=float, default=3.0)

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"[ERROR] Input directory does not exist: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if mda is None:
        sys.exit("[ERROR] MDAnalysis is required.")

    print("=" * 80)
    print("🚀 INITIATING ENHANCED POSE CURATION & POSEBUSTERS PIPELINE")
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
        if single_ref_mol:
            try:
                Chem.SanitizeMol(single_ref_mol)
            except Exception:
                pass

    buster = PoseBusters(config="dock") if (PoseBusters and not args.skip_bust) else None
    summary_records = []

    print("\n" + "=" * 80)
    print(f"{'POSE FILENAME':<32} | {'CLASHES':<8} | {'RMSD (Å)':<8} | {'PB SCORE':<10} | {'FAILED CHECKS'}")
    print("-" * 80)

    for pose_path in pose_files:
        base_name = os.path.basename(pose_path)
        out_name_pdb = os.path.splitext(base_name)[0] + "_fixed.pdb"
        out_name_sdf = os.path.splitext(base_name)[0] + "_fixed.sdf"
        out_path_pdb = os.path.join(args.output_dir, out_name_pdb)
        out_path_sdf = os.path.join(args.output_dir, out_name_sdf)
        complex_path = os.path.join(args.output_dir, os.path.splitext(base_name)[0] + "_complex.pdb")
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
            if target_template_mol:
                try:
                    Chem.SanitizeMol(target_template_mol)
                except Exception:
                    pass
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
            lig_vdw = np.array([VDW_RADII.get(a.GetSymbol().upper(), 1.70) for a in fixed_mol.GetAtoms()])
            cutoff_matrix = (lig_vdw[:, None] + env_vdw[None, :]) - CLASH_TOLERANCE
            num_clashes = int(np.count_nonzero(cdist(fixed_mol.GetConformer().GetPositions(), env_coords) < cutoff_matrix))
            shift_rmsd = 0.0
            record["Complex_PDB_Path"] = "NOT_GENERATED"
        else:
            try:
                fixed_mol, shift_rmsd = openmm_ligand_only_minimize(
                    args.protein, mol, complex_path, out_path_pdb,
                    max_iterations=args.max_min_iterations,
                    platform_name=args.openmm_platform,
                    param_dir=args.heme_params_dir,
                    ligand_charge=args.ligand_charge,
                    axial_info=axial_info,
                    workdir=workdir,
                    antechamber_charge=args.antechamber_charge
                )

                w = Chem.SDWriter(out_path_sdf)
                w.write(fixed_mol)
                w.close()

                lig_vdw = np.array([VDW_RADII.get(a.GetSymbol().upper(), 1.70) for a in fixed_mol.GetAtoms()])
                cutoff_matrix = (lig_vdw[:, None] + env_vdw[None, :]) - CLASH_TOLERANCE
                num_clashes = int(np.count_nonzero(
                    cdist(fixed_mol.GetConformer().GetPositions(), env_coords) < cutoff_matrix
                ))
                record["Complex_PDB_Path"] = complex_path

                if workdir.exists():
                    shutil.rmtree(workdir)

            except Exception as e:
                err_msg = str(e).splitlines()[0][:55]
                print(f"{base_name:<32} | {'N/A':<8} | {'N/A':<8} | {'ERROR':<10} | {err_msg}")
                record["PB_Score"] = "ERROR"
                summary_records.append(record)
                continue

        record["Remaining_Clashes"] = num_clashes
        record["Shift_RMSD_A"] = round(shift_rmsd, 3)

        if buster is not None:
            try:
                ref_eval_path = str(template_file) if template_file else (args.ref if args.ref and os.path.isfile(args.ref) else None)
                # Bust the output SDF with explicit Kekule bond orders
                df_pb = buster.bust(mol_pred=out_path_sdf, mol_cond=args.protein, mol_true=ref_eval_path)
                bool_cols = [c for c in df_pb.columns if df_pb[c].dtype == bool]
                total_tests = len(bool_cols)
                passed_tests = int(sum(df_pb[c].values[0] for c in bool_cols))
                failed_tests = [c for c in bool_cols if not df_pb[c].values[0]]

                pb_score_str = f"{passed_tests}/{total_tests}"
                failed_str = ", ".join(failed_tests) if failed_tests else "ALL PASSED"

                record["PB_Passed"] = passed_tests
                record["PB_Total"] = total_tests
                record["PB_Pass_Rate"] = round((passed_tests / total_tests) * 100, 1)
                record["Failed_Tests"] = "; ".join(failed_tests)

                for c in bool_cols:
                    record[f"PB_{c}"] = df_pb[c].values[0]

                print(f"{base_name:<32} | {num_clashes:<8} | {shift_rmsd:<8.3f} | {pb_score_str:<10} | {failed_str}")

            except Exception as e:
                err_msg = str(e).splitlines()[0][:50]
                print(f"{base_name:<32} | {num_clashes:<8} | {shift_rmsd:<8.3f} | {'ERROR':<10} | PB Exception: {err_msg}")
                record["PB_Score"] = "ERROR"
        else:
            print(f"{base_name:<32} | {num_clashes:<8} | {shift_rmsd:<8.3f} | {'SAVED':<10} | Done")

        summary_records.append(record)

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        csv_out = os.path.join(args.output_dir, "batch_posebusters_summary.csv")
        df_summary.to_csv(csv_out, index=False)
        print("=" * 80)
        print(f"\n[DONE] Batch processing complete across {len(summary_records)} poses.")
        print(f"• Fixed poses (.pdb and .sdf) saved to : {args.output_dir}/")
        print(f"• Summary Table saved to               : {csv_out}\n")


if __name__ == "__main__":
    main()