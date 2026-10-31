================================================================================
 VADOCKER GPU-ACCELERATED DOCKING & AMBER/OPENMM POSE CURATION PIPELINE
================================================================================

This package provides an end-to-end computational pipeline for GPU-accelerated 
molecular docking (VADocker), robust protein-ligand pose curation using AmberTools 
(tleap) and OpenMM, automated validation via PoseBusters, and RMSD/conformation analysis.

--------------------------------------------------------------------------------
 1. ATTACHED SCRIPTS & UTILITIES
--------------------------------------------------------------------------------
- dock_parallel1_gpu.py :
  GPU-accelerated batch docking drivers using PyTorch matrix-decomposition 
  kernels and empirical/AMBER free energy scoring.

- core_engine_empirical_gpu.py :
  Core docking engines containing conformer generation (ETKDGv3), grid search, 
  free energy scoring functions, and Powell microgrid local optimization.

- pose_curator_cypware_tleap_openmm4.py:
  Batch pose curator and PoseBusters validator. Parameterizes ligands via 
  Antechamber/Parmchk2 (GAFF2), builds Amber complex topologies with split-heme 
  conventions (CM1 axial cysteine, HM1 porphyrin ring, FE1 catalytic iron), 
  and minimizes ONLY ligand atoms in OpenMM while keeping the receptor and heme 
  system strictly frozen (Mass = 0.0 Da).

- merge-lig-conformation-pdb.py (merge_ligions.py):
  Merges two alternative ligand conformations (e.g., dual poses) from separate PDB 
  complexes into a single PDB file, applying 'A' and 'B' atom name suffixes, 
  0.50 occupancies, and full CONECT connectivity records.

- multi_rmsd_excel.py:
  Calculates unaligned ligand heavy-atom RMSDs across docked conformer batches using 
  RDKit topology matching and outputs a styled Excel summary report (.xlsx).

- generate_env_installer.py & restore_env.sh:
  Conda environment backup and automated restoration utilities.

--------------------------------------------------------------------------------
 2. ENVIRONMENT SETUP & INSTALLATION
--------------------------------------------------------------------------------
To replicate the exact software environment (vadocker_amber):

  1. Ensure Conda/Miniconda is installed on your target system.
  2. Run the automated restoration script:
     ./restore_env.sh
  3. Activate the newly created environment:
     conda activate vadocker_amber

(Alternatively, build manually using environment.yml: 
  conda env create -f environment.yml)

--------------------------------------------------------------------------------
 3. STEP-BY-STEP USAGE GUIDE
--------------------------------------------------------------------------------

Step 1: Run GPU-Accelerated Docking
--------------------------------------------------------------------------------
Execute the batch docking driver with your target receptor and ligand library:

  python -u dock_parallel1.py \
    --receptor 3NXU_align1.pdb \
    --ligands mol2_files \
    --references extracted_reference_ligands \
    --out drydock_output \
    --center 54.72 77.52 10.50 \
    --size 16.0 \
    --translation-step 1.0 \
    --rotation-step 10.0 \
    --max-conformers 50 \
    --energy-window 30.0 \
    --max-clashes 1 \
    --top-poses 10


Step 2: Pose Curation, OpenMM Minimization & PoseBusters Validation
--------------------------------------------------------------------------------
Relieve steric clashes inside the protein pocket while keeping the receptor, 
heme, iron, and axial cysteine rigidly frozen (Mass = 0.0 Da):

  python -u pose_curator_cypware_tleap_openmm4.py \
    --input-dir drydock_output/6MA8_sdf \
    --output-dir OCNT-fixed \
    -p 3NXU_min1.pdb \
    --ligands-dir mol2_files \
    --heme-params-dir ./CYPWare_HEME_parameters \
    --openmm-platform CUDA \
    --max-min-iterations 2000 \
    --ligand-charge 0 \



Step 3: Merge Alternative Conformations (Optional)
--------------------------------------------------------------------------------
Merge two distinct ligand conformations into a dual-occupancy PDB complex 
with preserved connectivity:

  python merge-lig-conformation-pdb.py \
    --complex1 complex1.pdb \
    --complex2 complex2.pdb \
    --output merged_dual_conformation.pdb \
    --ligand-resname UNL


Step 4: Batch RMSD & Conformation Analysis
--------------------------------------------------------------------------------
Evaluate heavy-atom RMSDs against a reference structure across all generated poses:

  python multi_rmsd_excel.py
  (Follow interactive prompts to enter reference PDB filename and wildcard pattern)

--------------------------------------------------------------------------------
 4. OUTPUT FILE STRUCTURE
--------------------------------------------------------------------------------
- drydock_output/<ligand_name>/ : Ranked docking PDB poses and local scores.csv.
- OCNT-fixed/                   : Curated SDF files, ps003-compatible fixed PDBs, 
                                  and batch_posebusters_summary.csv.
================================================================================