# VADocker
Virtual screening made Accessible with Docking (VADocker) is a new docking software that uses ligand translation, rotation, into binding sites and estimation of interactions.

🛠️ Installation & Environment Setup
VADocker requires an Anaconda/Miniconda environment.

1. Base Environment (All Platforms)
Create the environment and install the core structural dependency (RDKit):
conda create -n vadocker_env python=3.10
conda activate vadocker_env
conda install -c conda-forge rdkit obabel

For MacOS/Linux start the VADocker GUI with following command

python dock_gui3.py

💻 Command Line Interface (CLI) UsageFor batch processing on compute clusters or automated shell scripts, you can bypass the GUI and invoke the backend engines directly.
Global CLI ArgumentsArgumentDescriptionDefault--receptorPath to the target protein PDB file (Required).---ligandPath to a single ligand file (.pdb, .sdf, .mol2).---ligandsPath to a directory for batch processing multiple ligands.---outDestination directory for output files and CSV summaries.---centerActive site geometric center (X Y Z).---sizeSearch grid box size in Ångströms.20.0--translation-stepGrid node spatial resolution in Ångströms.1.5--rotation-stepAngular rotation step size in degrees.20.0--scoringActive scoring engine (empirical, amber, gnina_ml).empirical--top-posesNumber of top hit poses to save per ligand.10--coresNumber of CPU cores to dedicate to the worker pool. All available

Mac / Linux (Empirical or Amber Mode)
Using the Unix-optimized engine (dock_parallel1.py) to process an entire directory of ligands, utilizing empirical scoring final pose ranking.

Bash command for docking multiple ligands

python dock_parallel1.py \
  --receptor "/Users/Workspace/Receptors/5TE8-A.pdb" \
  --ligands "/Users/Workspace/Ligand_Library/" \
  --out "/Users/Workspace/Results/batch_run_01" \
  --center 11.992 -0.380 -52.714 \
  --size 20.0 \
  --translation-step 2.0 \
  --rotation-step 15.0 \
  --top-poses 5 \
  --scoring empirical \
  --cores 8

  Bash command for single ligand docking
  python dock_parallel1.py \
  --receptor "/Path/To/receptor.pdb" \
  --ligand "/Path/To/your_ligand.pdb" \
  --out "/Path/To/Output_Folder" \
  --center 11.992 -0.380 -52.714 \
  --size 15.0 \
  --translation-step 2.0 \
  --rotation-step 20.0 \
  --top-poses 10 \
  --scoring empirical \
  --cores 4

  

  📜 Academic Declaration & Acknowledgments
AI Assistance Statement:
During the development of this repository, the authors utilized Large Language Models (including Google Gemini) to assist in the generation of parallel multiprocessing architecture, GUI boilerplate construction, and cross-platform error handling. All generated logic was thoroughly reviewed, modified, and validated by the authors to ensure scientific accuracy and computational robustness.
