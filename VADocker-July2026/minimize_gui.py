#!/usr/bin/env python3
import sys
import os
import csv
import threading
import subprocess
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Re-use your backend functions inside the GUI thread wrapper context
from minimize_parallel import (
    detect_amber_binary, 
    get_mol2_residue_name, 
    force_residue_name_in_pdb, 
    run_quick_minimization
)

class VADockerMinimizerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VADocker - AMBER Complex Minimization & Structural Clash Smoother")
        self.geometry("1020x850")
        self.minsize(950, 750)
        
        # Light Blue Workstation Theme Palette Configuration
        self.bg_main = "#EBF3FA"       # Crisp Soft Light Blue (Window Canvas)
        self.bg_panel = "#FFFFFF"      # Pure White (Structured Entry Panels)
        self.bg_input = "#F1F7FC"      # Clean Slate-Tint Input Backdrops
        self.bg_terminal = "#111622"   # Deep Obsidian Black (Console Output Matrix)
        
        self.fg_dark = "#1E293B"       # Dark Corporate Slate for High-Contrast Text
        self.fg_muted = "#64748B"      # Secondary Muted Info Labels
        self.fg_terminal = "#E2E8F0"   # Terminal Screen Text Off-White
        
        self.accent_blue = "#2563EB"   # High-Visibility Primary Theme Blue
        self.accent_green = "#10B981"  # Clinical Validation Success Green
        self.accent_red = "#EF4444"    # Structural Anomaly Warning Red
        self.accent_border = "#CBD5E1" # Clear Panel Edge Separation Grid Gray
        
        self.configure(bg=self.bg_main)
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure('.', bg=self.bg_main, foreground=self.fg_dark)
        style.configure('TFrame', background=self.bg_main)
        style.configure('Panel.TFrame', background=self.bg_panel, relief="solid", borderwidth=1, bordercolor=self.accent_border)
        
        style.configure('TLabel', background=self.bg_main, foreground=self.fg_dark, font=("Helvetica", 10, "bold"))
        style.configure('Panel.TLabel', background=self.bg_panel, foreground=self.fg_dark, font=("Helvetica", 10, "bold"))
        style.configure('Header.TLabel', background=self.bg_panel, foreground=self.accent_blue, font=("Helvetica", 12, "bold"))
        style.configure('Muted.TLabel', background=self.bg_panel, foreground=self.fg_muted, font=("Helvetica", 9, "italic"))
        
        style.configure('TEntry', fieldbackground=self.bg_input, foreground=self.fg_dark, insertcolor=self.fg_dark)
        style.configure('TCombobox', fieldbackground=self.bg_input, background=self.bg_panel, foreground=self.fg_dark)
        
        style.configure('Action.TButton', background=self.accent_blue, foreground="#FFFFFF", font=("Helvetica", 11, "bold"))
        style.map('Action.TButton', background=[('active', '#1D4ED8'), ('disabled', '#94A3B8')])
        
        style.configure('Browse.TButton', background="#E2E8F0", foreground=self.fg_dark, font=("Helvetica", 10))
        style.map('Browse.TButton', background=[('active', '#CBD5E1')])
        
        style.configure('Demo.TButton', background="#059669", foreground="#FFFFFF", font=("Helvetica", 10, "bold"))
        style.map('Demo.TButton', background=[('active', '#047857')])
        
        style.configure('TRadiobutton', background=self.bg_panel, foreground=self.fg_dark, font=("Helvetica", 10, "bold"))
        style.map('TRadiobutton', background=[('active', self.bg_panel)], foreground=[('active', self.fg_dark)])

    def create_widgets(self):
        # Header banner block
        title_frame = tk.Frame(self, bg=self.accent_blue, height=55, relief="flat")
        title_frame.pack(fill="x", side="top", padx=0, pady=0)
        
        title_lbl = tk.Label(title_frame, text="🧬 VADOCKER - AMBER COMPLEX MINIMIZER", bg=self.accent_blue, fg="#FFFFFF", font=("Helvetica", 13, "bold"))
        title_lbl.pack(side="left", padx=15, pady=12)
        
        version_lbl = tk.Label(title_frame, text="CYP450 Structural Refinement Deck | July-2026", bg=self.accent_blue, fg="#EBF3FA", font=("Helvetica", 10, "bold"))
        version_lbl.pack(side="right", padx=15, pady=15)

        # Dynamic Grid Container Configuration for Clean Resizing Behaviors
        main_container = ttk.Frame(self, padding="12")
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(1, weight=1)  # Allow terminal panel on right to scale horizontally
        main_container.rowconfigure(0, weight=1)     # Allow uniform vertical scale expansion

        # Left Column: Configuration Parameters Panel
        left_column = ttk.Frame(main_container, width=470)
        left_column.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        left_column.pack_propagate(False) # Frame size integrity guarantee lock
        
        # Input Data Panels Layout
        files_panel = ttk.Frame(left_column, style='Panel.TFrame', padding="14")
        files_panel.pack(fill="both", expand=True)
        ttk.Label(files_panel, text="📁 Complex Environment Components", style='Header.TLabel').pack(anchor="w", pady=(0, 10))
        
        self.receptor_path = self.create_file_row(files_panel, "Target Apo Receptor PDB File:", browse_dir=False, default_val="../../5TE8-A.pdb")
        
        mode_frame = ttk.Frame(files_panel, style='Panel.TFrame')
        mode_frame.pack(fill="x", pady=6)
        self.ligand_mode = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Pose Run Folder", variable=self.ligand_mode, value="single", command=self.toggle_mode_labels).pack(side="left", padx=(0, 24))
        ttk.Radiobutton(mode_frame, text="Batch Directory Matrix", variable=self.ligand_mode, value="batch", command=self.toggle_mode_labels).pack(side="left")
        
        self.dock_lbl_var = tk.StringVar(value="Ligand Docking Output Folder Path:")
        self.dock_dir_path = self.create_file_row(files_panel, self.dock_lbl_var, browse_dir=True, default_val="/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/REACCESS-validation/drug_multipleOH/dockdrugs/methyldopa_amber_fixed")
        
        ttk.Label(files_panel, text="🛠️ Forcefield Generation Parameters", style='Header.TLabel').pack(anchor="w", pady=(15, 6))
        
        self.leap_protein = self.create_input_row(files_panel, "LEaP Protein Forcefield parameters:", default_val="source leaprc.protein.ff14SB")
        self.leap_gaff = self.create_input_row(files_panel, "LEaP Ligand Forcefield parameters:", default_val="source leaprc.gaff")
        self.pb_radii = self.create_input_row(files_panel, "PBRadii Solvent Model Topology Map:", default_val="set default PBRadii mbondi2")
        
        # Right Column: Terminal Screen Monitors Matrix
        right_column = ttk.Frame(main_container, style='Panel.TFrame', padding="14")
        right_column.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        
        ttk.Label(right_column, text="🖥️ AmberTools Standard Output Monitor Logs", style='Header.TLabel').pack(anchor="w", pady=(0, 6))
        
        self.console = ScrolledText(right_column, bg=self.bg_terminal, fg=self.fg_terminal, insertbackground=self.fg_terminal, font=("Courier New", 10), state="disabled", bd=0, highlightthickness=1, highlightbackground=self.accent_border)
        self.console.pack(fill="both", expand=True, pady=5)
        
        btn_frame = ttk.Frame(right_column, style='Panel.TFrame')
        btn_frame.pack(fill="x", pady=(10, 0))
        
        self.btn_run = ttk.Button(btn_frame, text="⚡ START MINIMIZATION ENGINE", style='Action.TButton', command=self.start_minimization_thread)
        self.btn_run.pack(side="right", padx=(12, 0), ipadx=12, ipady=5)
        
        self.btn_demo = ttk.Button(btn_frame, text="Load Demo Set", style='Demo.TButton', command=self.load_demo_values)
        self.btn_demo.pack(side="left", padx=(0, 10), ipady=3)
        
        self.btn_clear = ttk.Button(btn_frame, text="Clear Logs", style='Browse.TButton', command=self.clear_console)
        self.btn_clear.pack(side="left", ipady=3)

    def create_file_row(self, parent, label_text, browse_dir=False, default_val=""):
        frame = ttk.Frame(parent, style='Panel.TFrame')
        frame.pack(fill="x", pady=6)
        
        if isinstance(label_text, tk.StringVar):
            label = ttk.Label(frame, textvariable=label_text, style='Panel.TLabel')
        else:
            label = ttk.Label(frame, text=label_text, style='Panel.TLabel')
        label.pack(anchor="w", pady=(0, 4))
        
        sub_frame = ttk.Frame(frame, style='Panel.TFrame')
        sub_frame.pack(fill="x")
        
        var = tk.StringVar(value=default_val)
        entry = tk.Entry(sub_frame, textvariable=var, bg=self.bg_input, fg=self.fg_dark, bd=1, relief="solid", highlightthickness=0, font=("Helvetica", 10))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
        
        cmd = lambda: self.browse_path(var, browse_dir=True if (browse_dir or self.ligand_mode.get() == "batch") else False)
        ttk.Button(sub_frame, text="Browse...", style='Browse.TButton', command=cmd).pack(side="right")
        return var

    def create_input_row(self, parent, label_text, default_val=""):
        frame = ttk.Frame(parent, style='Panel.TFrame')
        frame.pack(fill="x", pady=6)
        ttk.Label(frame, text=label_text, style='Panel.TLabel').pack(anchor="w", pady=(0, 4))
        var = tk.StringVar(value=default_val)
        entry = tk.Entry(frame, textvariable=var, bg=self.bg_input, fg=self.fg_dark, bd=1, relief="solid", highlightthickness=0, font=("Helvetica", 10))
        entry.pack(fill="x", ipady=3)
        return var

    def toggle_mode_labels(self):
        if self.ligand_mode.get() == "single":
            self.dock_lbl_var.set("Ligand Docking Output Folder Path:")
        else:
            self.dock_lbl_var.set("Directory for Docking Batch Matrices:")

    def browse_path(self, target_var, browse_dir=False):
        if browse_dir:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=[("PDB Structure File", "*.pdb"), ("All Files", "*.*")])
        if path:
            target_var.set(path)

    def load_demo_values(self):
        self.receptor_path.set("../../5TE8-A.pdb")
        self.ligand_mode.set("single")
        self.toggle_mode_labels()
        self.dock_dir_path.set("/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/REACCESS-validation/drug_multipleOH/dockdrugs/methyldopa_amber_fixed")
        self.leap_protein.set("source leaprc.protein.ff14SB")
        self.leap_gaff.set("source leaprc.gaff")
        self.pb_radii.set("set default PBRadii mbondi2")
        self.log_message("[DEMO INITIALIZED] Parameters mapped and consistent for subsequent SOM/BDE downstream usage.\n", "#00FF66")

    def log_message(self, text, color=None):
        self.console.config(state="normal")
        start_idx = self.console.index("end-1c")
        self.console.insert("end", text)
        end_idx = self.console.index("end-1c")
        
        if color:
            tag_name = f"color_{color}"
            self.console.tag_config(tag_name, foreground=color)
            self.console.tag_add(tag_name, start_idx, end_idx)
        else:
            if "SUCCESS" in text or "Cleaned Delta E" in text:
                self.console.tag_config("neon_green", foreground="#00FF66", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_green", start_idx, end_idx)
            elif "Error" in text or "CRITICAL" in text or "[✗]" in text:
                self.console.tag_config("neon_red", foreground="#FF3B30", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_red", start_idx, end_idx)
            elif "🚀" in text or "Relaxing Structures" in text:
                self.console.tag_config("neon_blue", foreground="#60A5FA", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_blue", start_idx, end_idx)
                
        self.console.see("end")
        self.console.config(state="disabled")

    def clear_console(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def start_minimization_thread(self):
        if not Path(self.receptor_path.get()).is_file():
            messagebox.showerror("Path Resolution Error", "The specified Apo Receptor PDB structural map does not exist.")
            return
        if not Path(self.dock_dir_path.get()).is_dir():
            messagebox.showerror("Path Resolution Error", "The specified Ligand output execution folder could not be located.")
            return
            
        self.btn_run.config(state="disabled")
        self.clear_console()
        threading.Thread(target=self.run_minimization_core, daemon=True).start()

    def process_single_folder(self, dock_folder: Path, receptor: Path, amber_bin: str):
        """Processes an individual docking workspace folder cleanly."""
        l_mol2 = dock_folder / "reusable_ligand.mol2"
        l_frcmod = dock_folder / "reusable_ligand.frcmod"
        scores_csv = dock_folder / "scores.csv"

        if not scores_csv.is_file():
            self.log_message(f"[✗] Error: Missing scores.csv inside directory: {dock_folder.name}\n", "#FF3B30")
            return

        rows = []
        with scores_csv.open("r") as f:
            reader = csv.DictReader(f)
            for r in reader: 
                rows.append(r)

        if not rows:
            return

        # Check for placeholder topology parameters
        is_placeholder = False
        if l_mol2.is_file(): 
            is_placeholder = "Placeholder" in l_mol2.read_text()
        
        if not l_mol2.is_file() or is_placeholder:
            first_pose_path = Path(rows[0]["output_file"])
            if not first_pose_path.is_file():
                first_pose_path = dock_folder / first_pose_path.name
                
            self.log_message(f" -> [Antechamber Parameter Check] Generating missing parameters for {dock_folder.name}...\n", "#60A5FA")
            subprocess.run(["antechamber", "-i", str(first_pose_path), "-fi", "pdb", "-o", str(l_mol2), "-fo", "mol2", "-c", "bcc", "-s", "2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["parmchk2", "-i", str(l_mol2), "-f", "mol2", "-o", str(l_frcmod)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        mol2_res = get_mol2_residue_name(l_mol2)
        cached_e_rec = None
        rescored_rows = []

        for idx, row in enumerate(rows, start=1):
            pose_pdb = Path(row["output_file"])
            if not pose_pdb.is_file():
                pose_pdb = dock_folder / pose_pdb.name

            self.log_message(f"[{idx}/{len(rows)}] Relaxing Structures for: {pose_pdb.name}\n")
            force_residue_name_in_pdb(pose_pdb, mol2_res)
            
            pose_amber_dir = pose_pdb.parent / f"pose_{idx:03d}_quick_workspace"
            pose_amber_dir.mkdir(parents=True, exist_ok=True)
            
            prmtop_com = pose_amber_dir / "complex.prmtop"
            inpcrd_com = pose_amber_dir / "complex.inpcrd"
            prmtop_rec = pose_amber_dir / "receptor_apo.prmtop"
            inpcrd_rec = pose_amber_dir / "receptor_apo.inpcrd"
            prmtop_lig = pose_amber_dir / "ligand_v.prmtop"
            inpcrd_lig = pose_amber_dir / "ligand_v.inpcrd"
            
            tleap_in = pose_amber_dir / "tleap_quick.in"
            tleap_content = f"""
{self.leap_protein.get()}
{self.leap_gaff.get()}
{self.pb_radii.get()}
loadAmberParams {l_frcmod.resolve()}
{mol2_res} = loadmol2 {l_mol2.resolve()}

RECEP = loadpdb {receptor.resolve()}
LIG = loadpdb {pose_pdb.resolve()}
COM = combine {{ RECEP LIG }}

saveamberparm COM {prmtop_com.resolve()} {inpcrd_com.resolve()}
saveamberparm LIG {prmtop_lig.resolve()} {inpcrd_lig.resolve()}
"""
            if cached_e_rec is None:
                tleap_content += f"saveamberparm RECEP {prmtop_rec.resolve()} {inpcrd_rec.resolve()}\n"
                
            tleap_content += "quit\n"
            tleap_in.write_text(tleap_content)
            subprocess.run(["tleap", "-f", str(tleap_in)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not (prmtop_com.is_file() and prmtop_lig.is_file()):
                row["e_complex"] = "999.9"
                row["e_receptor_apo"] = "999.9"
                row["e_ligand"] = "999.9"
                row["delta_E_binding"] = "999.9"
                rescored_rows.append(row)
                continue

            if cached_e_rec is None:
                cached_e_rec = run_quick_minimization(amber_bin, prmtop_rec, inpcrd_rec, pose_amber_dir, "receptor")
            
            e_rec = cached_e_rec
            e_com = run_quick_minimization(amber_bin, prmtop_com, inpcrd_com, pose_amber_dir, "complex")
            e_lig = run_quick_minimization(amber_bin, prmtop_lig, inpcrd_lig, pose_amber_dir, "ligand")

            if 999.9 in (e_com, e_rec, e_lig):
                e_bind = 999.9
            else:
                e_bind = e_com - (e_rec + e_lig)
                
            self.log_message(f"  • Minimized Energies: E_com={e_com:.1f} | E_rec={e_rec:.1f} | E_lig={e_lig:.1f}\n")
            self.log_message(f"  • Cleaned Delta E Binding: {e_bind:.3f} kcal/mol\n")
            
            row["e_complex"] = f"{e_com:.3f}"
            row["e_receptor_apo"] = f"{e_rec:.3f}"
            row["e_ligand"] = f"{e_lig:.3f}"
            row["delta_E_binding"] = f"{e_bind:.3f}"
            
            rescored_rows.append(row)

        out_csv_path = dock_folder / "scores_amber_min.csv"
        new_fields = ["e_complex", "e_receptor_apo", "e_ligand", "delta_E_binding"]
        
        fieldnames = list(rows[0].keys())
        for field in new_fields:
            if field not in fieldnames: 
                fieldnames.append(field)

        with out_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rescored_rows)

        self.log_message(f"[✔] SUCCESS: Finished optimization report -> {out_csv_path.name}\n", "#00FF66")

    def run_minimization_core(self):
        self.log_message("="*60 + "\n")
        self.log_message("🚀 QUICK-MIN AUTOMATION: STERIC CLASH RELAXATION MATRIX RUN\n")
        self.log_message("="*60 + "\n")

        amber_bin = detect_amber_binary()
        if not amber_bin:
            self.log_message("[✗] Error: No local execution instance of Amber located on your system path.\n", "#FF3B30")
            self.btn_run.config(state="normal")
            return
        
        self.log_message(f" -> Execution Layer Discovered: '{amber_bin}'\n")
        
        receptor = Path(self.receptor_path.get())
        dock_target = Path(self.dock_dir_path.get())
        
        t_start = time.time()
        
        if self.ligand_mode.get() == "single":
            self.process_single_folder(dock_target, receptor, amber_bin)
        else:
            # Batch Directory scanning handles folder trees seamlessly
            self.log_message(f" -> Batch Scanning directory pathways under grid targets...\n")
            subdirs = sorted([d for d in dock_target.iterdir() if d.is_dir()])
            if not subdirs:
                self.log_message("[✗] Error: No subdirectories discovered inside target path.\n", "#FF3B30")
            for folder in subdirs:
                self.log_message(f"\n📂 Entering workspace node: {folder.name}\n", "#60A5FA")
                self.process_single_folder(folder, receptor, amber_bin)
                
        self.log_message(f"\n🎉 Refinement cycle completed. Time taken: {time.time() - t_start:.2f}s\n", "#00FF66")
        self.btn_run.config(state="normal")

if __name__ == "__main__":
    app = VADockerMinimizerGUI()
    app.mainloop()