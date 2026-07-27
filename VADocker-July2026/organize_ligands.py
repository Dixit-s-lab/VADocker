#!/usr/bin/env python3
import shutil
from pathlib import Path

def consolidate_and_rename_ligands():
    # Define active folder layouts relative to your working directory
    current_dir = Path.cwd()
    dest_dir = current_dir / "ligands-for-docking"
    
    # Initialize target directory if it doesn't exist yet
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print(f"🧬 LIGAND WORKSPACE CONSOLIDATION ENGINE")
    print(f" -> Destination Node: {dest_dir.name}/")
    print("=" * 65)
    
    moved_count = 0
    
    # Recursively scan all subdirectories for your target structural profiles
    for file_path in current_dir.rglob("*_amber_fixed.pdb"):
        # Prevent picking up files that are already inside the destination directory
        if dest_dir in file_path.parents:
            continue
            
        # Clean naming mapping strategy: isolate the clean prefix base string
        old_name = file_path.name
        new_name = old_name.replace("_amber_fixed.pdb", ".pdb")
        target_dest = dest_dir / new_name
        
        try:
            # Copy file to destination with the clean, condensed layout naming scheme
            shutil.copy2(file_path, target_dest)
            moved_count += 1
            print(f" [✔] Transferred: {file_path.parent.name}/{old_name} -> {new_name}")
        except Exception as e:
            print(f" [✗] Error transferring {old_name}: {str(e)}")
            
    print("=" * 65)
    print(f"🎉 Array processing completed successfully.")
    print(f" -> Consolidated {moved_count} clean structural targets into '{dest_dir.name}/'.\n")

if __name__ == "__main__":
    consolidate_and_rename_ligands()