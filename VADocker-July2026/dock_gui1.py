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

class VADockerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VADocker - High-Performance Parallel Lattice Search Suite")
        self.geometry("1020x850")
        self.minsize(950, 750)
        
        # Professional Light Blue Workstation Palette
        self.bg_main = "#EBF3FA"       
        self.bg_panel = "#FFFFFF"      
        self.bg_input = "#F1F7FC"      
        self.bg_terminal = "#111622"   
        self.fg_dark = "#1E293B"       
        self.fg_muted = "#64748B"      
        self.fg_terminal = "#E2E8F0"   
        self.accent_blue = "#2563EB"   
        self.accent_green = "#10B981"  
        self.accent_red = "#EF4444"    
        self.accent_border = "#CBD5E1" 
        
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

    def create_widgets(self):
        title_frame = tk.Frame(self, bg=self.accent_blue, height=55, relief="flat")
        title_frame.pack(fill="x", side="top", padx=0, pady=0)
        
        title_lbl = tk.Label(title_frame, text="🧬 VADOCKER MATRIX PARALLEL SUITE", bg=self.accent_blue, fg="#FFFFFF", font=("Helvetica", 13, "bold"))
        title_lbl.pack(side="left", padx=15, pady=12)
        
        version_lbl = tk.Label(title_frame, text="Workstation Environment Core | July-2026", bg=self.accent_blue, fg="#EBF3FA", font=("Helvetica", 10, "bold"))
        version_lbl.pack(side="right", padx=15, pady=15)

        main_container = ttk.Frame(self, padding="12")
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(1, weight=1)  
        main_container.rowconfigure(0, weight=1)     

        left_column = ttk.Frame(main_container, width=470)
        left_column.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        left_column.pack_propagate(False)
        
        files_panel = ttk.Frame(left_column, style='Panel.TFrame', padding="14")
        files_panel.pack(fill="x", pady=(0, 10))
        ttk.Label(files_panel, text="📁 File System Assignments", style='Header.TLabel').pack(anchor="w", pady=(0, 10))
        
        self.receptor_path = self.create_file_row(files_panel, "Target Receptor PDB File:", browse_dir=False, default_val="../../5TE8-A.pdb")
        
        mode_frame = ttk.Frame(files_panel, style='Panel.TFrame')
        mode_frame.pack(fill="x", pady=6)
        self.ligand_mode = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Target Ligand", variable=self.ligand_mode, value="single", command=self.toggle_ligand_inputs).pack(side="left", padx=(0, 24))
        ttk.Radiobutton(mode_frame, text="Batch Directory Target Matrix", variable=self.ligand_mode, value="batch", command=self.toggle_ligand_inputs).pack(side="left")
        
        self.ligand_lbl_var = tk.StringVar(value="Ligand Molecule File for Docking:")
        self.ligand_path = self.create_file_row(files_panel, self.ligand_lbl_var, browse_dir=False, default_val="/Users/dr.vaibhava.dixit/Library/CloudStorage/GoogleDrive-vaibhav@niperguwahati.in/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/amber_parameters/methyldopa/methyldopa_amber_fixed.pdb")
        
        self.reference_path = self.create_file_row(files_panel, "Validation Reference PDB File (Optional):", browse_dir=False, default_val="/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/MIDA-refA.pdb")
        self.smiles_str = self.create_input_row(files_panel, "Enforce Structural SMILES Template String:")
        self.out_path = self.create_file_row(files_panel, "Results Destination Workspace Directory Folder:", browse_dir=True, default_val="/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/REACCESS-validation/drug_multipleOH/dockdrugs")

        params_panel = ttk.Frame(left_column, style='Panel.TFrame', padding="14")
        params_panel.pack(fill="both", expand=True)
        ttk.Label(params_panel, text="🎛️ Binding Pocket Grid Variables & Precision Matrix", style='Header.TLabel').pack(anchor="w", pady=(0, 10))
        
        grid_coord_frame = ttk.Frame(params_panel, style='Panel.TFrame')
        grid_coord_frame.pack(fill="x", pady=5)
        
        ttk.Label(grid_coord_frame, text="Geometric Center Coords (X, Y, Z):", style='Panel.TLabel').pack(anchor="w", pady=(0, 4))
        coords_input_frame = ttk.Frame(grid_coord_frame, style='Panel.TFrame')
        coords_input_frame.pack(fill="x", pady=2)
        
        self.center_x = tk.StringVar(value="11.992")
        self.center_y = tk.StringVar(value="-0.380")
        self.center_z = tk.StringVar(value="-52.714")
        
        for var in [self.center_x, self.center_y, self.center_z]:
            tk.Entry(coords_input_frame, textvariable=var, bg=self.bg_input, fg=self.fg_dark, bd=1, relief="solid", highlightthickness=0, font=("Helvetica", 10), width=11).pack(side="left", padx=(0, 12))
        
        self.box_size = self.create_input_row(params_panel, "Active Search Grid Dimension Box Size (Å):", default_val="15.0")
        self.trans_step = self.create_input_row(params_panel, "Grid Node Box Translation Resolution Step Size (Å):", default_val="2.0")
        self.rot_step = self.create_input_row(params_panel, "Sampling Orientation Density Angular Rotation Step (Deg):", default_val="20.0")
        
        filter_frame = ttk.Frame(params_panel, style='Panel.TFrame')
        filter_frame.pack(fill="x", pady=8)
        
        left_sub = ttk.Frame(filter_frame, style='Panel.TFrame')
        left_sub.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Label(left_sub, text="Top Saved Hit Poses Limit:", style='Panel.TLabel').pack(anchor="w")
        self.top_poses = tk.StringVar(value="10")
        tk.Entry(left_sub, textvariable=self.top_poses, bg=self.bg_input, fg=self.fg_dark, bd=1, relief="solid", highlightthickness=0, font=("Helvetica", 10)).pack(fill="x", pady=4)
        
        right_sub = ttk.Frame(filter_frame, style='Panel.TFrame')
        right_sub.pack(side="right", fill="x", expand=True, padx=(6, 0))
        ttk.Label(right_sub, text="Distributed Worker Core Count:", style='Panel.TLabel').pack(anchor="w")
        self.core_count = tk.StringVar(value="4")
        tk.Entry(right_sub, textvariable=self.core_count, bg=self.bg_input, fg=self.fg_dark, bd=1, relief="solid", highlightthickness=0, font=("Helvetica", 10)).pack(fill="x", pady=4)
        
        score_frame = ttk.Frame(params_panel, style='Panel.TFrame')
        score_frame.pack(fill="x", pady=6)
        ttk.Label(score_frame, text="Active Forcefield Engine Potential Function Selection Routine:", style='Panel.TLabel').pack(anchor="w", pady=(0, 4))
        self.scoring_mode = tk.StringVar(value="empirical")
        ttk.Combobox(score_frame, textvariable=self.scoring_mode, values=["empirical", "amber"], state="readonly").pack(fill="x", pady=2)

        right_column = ttk.Frame(main_container, style='Panel.TFrame', padding="14")
        right_column.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        
        ttk.Label(right_column, text="🖥️ Console Stream Log Output Monitor", style='Header.TLabel').pack(anchor="w", pady=(0, 6))
        
        self.console = ScrolledText(right_column, bg=self.bg_terminal, fg=self.fg_terminal, insertbackground=self.fg_terminal, font=("Courier New", 10), state="disabled", bd=0, highlightthickness=1, highlightbackground=self.accent_border)
        self.console.pack(fill="both", expand=True, pady=5)
        
        btn_frame = ttk.Frame(right_column, style='Panel.TFrame')
        btn_frame.pack(fill="x", pady=(10, 0))
        
        self.btn_run = ttk.Button(btn_frame, text="⚡ START RUN DOCKING LOOP", style='Action.TButton', command=self.run_docking_thread)
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
        
        cmd = lambda: self.browse_path(var, browse_dir)
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

    def toggle_ligand_inputs(self):
        if self.ligand_mode.get() == "single":
            self.ligand_lbl_var.set("Ligand Molecule File for Docking:")
        else:
            self.ligand_lbl_var.set("Directory for Docking:")

    def browse_path(self, target_var, is_dir=False):
        if is_dir or self.ligand_mode.get() == "batch" and target_var == self.ligand_path:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=[("Molecular Structure Formats", "*.pdb *.mol2 *.sdf"), ("All Files", "*.*")])
        if path:
            target_var.set(path)

    def load_demo_values(self):
        self.receptor_path.set("../../5TE8-A.pdb")
        self.ligand_mode.set("single")
        self.toggle_ligand_inputs()
        self.ligand_path.set("/Users/dr.vaibhava.dixit/Library/CloudStorage/GoogleDrive-vaibhav@niperguwahati.in/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/amber_parameters/methyldopa/methyldopa_amber_fixed.pdb")
        self.reference_path.set("/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/MIDA-refA.pdb")
        self.out_path.set("/Users/dr.vaibhava.dixit/Google Drive/My Drive/NIPER-Guwahati/drug-metabolism/disangle-dock-MD.py/REACCESS-validation/drug_multipleOH/dockdrugs")
        self.center_x.set("11.992")
        self.center_y.set("-0.380")
        self.center_z.set("-52.714")
        self.box_size.set("15.0")
        self.trans_step.set("2.0")
        self.rot_step.set("20.0")
        self.top_poses.set("10")
        self.core_count.set("4")
        self.scoring_mode.set("empirical")
        self.smiles_str.set("")
        self.log_message("[DEMO PRESETS LOADED] Global composite scoring engine ready.\n", "#00FF66")

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
            if "SUCCESS" in text or "✔" in text:
                self.console.tag_config("neon_green", foreground="#00FF66", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_green", start_idx, end_idx)
            elif "CRITICAL" in text or "FAIL" in text or "⚠️" in text:
                self.console.tag_config("neon_red", foreground="#FF3B30", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_red", start_idx, end_idx)
            elif "📊" in text or "==================" in text:
                self.console.tag_config("neon_blue", foreground="#60A5FA", font=("Courier New", 10, "bold"))
                self.console.tag_add("neon_blue", start_idx, end_idx)
            elif "•" in text:
                self.console.tag_config("bright_white", foreground="#FFFFFF")
                self.console.tag_add("bright_white", start_idx, end_idx)
                
        self.console.see("end")
        self.console.config(state="disabled")

    def clear_console(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def run_docking_thread(self):
        if not self.receptor_path.get() or not self.ligand_path.get() or not self.out_path.get():
            messagebox.showerror("Param Error", "Critical setup pathways are blank! Confirm Receptor, Ligand, and Output targets.")
            return
        self.btn_run.config(state="disabled")
        self.clear_console()
        threading.Thread(target=self.execute_docking_process, daemon=True).start()

    def parse_and_print_summary(self, ligand_file_path):
        ligand_name = Path(ligand_file_path).stem
        output_dir = Path(self.out_path.get()) / ligand_name
        csv_path = output_dir / "scores.csv"
        
        time.sleep(0.3) 
        if not csv_path.is_file():
            return

        try:
            rows = []
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
                        
            if not rows:
                return

            best_row = rows[0]

            self.log_message("\n" + "="*55 + "\n")
            self.log_message(f"📊 EQUAL-WEIGHT COMPOSITE SUMMARY: {Path(ligand_file_path).name}\n")
            self.log_message("="*55 + "\n")
            self.log_message(f"  • Top Pose Rank Assignment      : {best_row.get('pose_rank', 'N/A')}\n")
            self.log_message(f"  • Top Pose File Name            : {best_row.get('output_file', 'N/A')}\n")
            self.log_message(f"  • Defined Master Score          : {best_row.get('score', 'N/A')} (Composite, lower is better)\n")
            self.log_message(f"  • Underlying Raw Potential      : {float(best_row.get('raw_score', best_row.get('score', 0.0))):.4f}\n")
            self.log_message(f"  • Steric Clashes Count          : {best_row.get('clashes', '0')}\n")
            self.log_message(f"  • Heavy Atom Contacts           : {best_row.get('contacts', '0')}\n")
            self.log_message(f"  • Hydrogen Bonds Count          : {best_row.get('hbonds', '0')}\n")
            
            rmsd_val = float(best_row.get("rmsd_to_reference", 0.0))
            if rmsd_val > 0.0:
                status_str = "SUCCESS" if rmsd_val <= 2.0 else "FAIL"
                status_color = "#00FF66" if status_str == "SUCCESS" else "#FF3B30"
                self.log_message(f"  • Reference RMSD                : {rmsd_val:.4f} Å [Verification: ")
                self.log_message(status_str, status_color)
                self.log_message("]\n")
            else:
                self.log_message("  • Reference RMSD                : N/A (Excluded from Weighting)\n")
                
            self.log_message(f"  • Total Saved Hit Poses         : {len(rows)} profiles\n")
            self.log_message("="*55 + "\n")
            
        except Exception as e:
            pass

    def execute_docking_process(self):
        script_dir = Path(__file__).parent.resolve()
        
        # TARGET SCRIPT UPDATED TO dock_parallel1.py
        driver_script = script_dir / "dock_parallel1.py"
        
        if not driver_script.is_file():
            self.log_message(f"\n[CRITICAL ERROR] The backend engine '{driver_script.name}' could not be found in the directory!\n", "#FF3B30")
            self.btn_run.config(state="normal")
            return
            
        cmd = [
            sys.executable, str(driver_script),
            "--receptor", self.receptor_path.get(),
            "--out", self.out_path.get(),
            "--center", self.center_x.get(), self.center_y.get(), self.center_z.get(),
            "--size", self.box_size.get(),
            "--translation-step", self.trans_step.get(),
            "--rotation-step", self.rot_step.get(),
            "--top-poses", self.top_poses.get(),
            "--scoring", self.scoring_mode.get(),
            "--cores", self.core_count.get()
        ]
        
        if self.ligand_mode.get() == "single":
            cmd.extend(["--ligand", self.ligand_path.get()])
            target_ligands = [self.ligand_path.get()]
        else:
            cmd.extend(["--ligands", self.ligand_path.get()])
            lp = Path(self.ligand_path.get())
            target_ligands = sorted(str(f) for f in lp.iterdir() if f.is_file() and f.suffix.lower() in {".pdb", ".mol2", ".sdf"}) if lp.is_dir() else []
            
        if self.reference_path.get().strip():
            cmd.extend(["--reference", self.reference_path.get()])
            
        if self.smiles_str.get().strip():
            cmd.extend(["--smiles", self.smiles_str.get().strip()])

        self.log_message(f"Initiating process array stream using command string layout:\n{' '.join(cmd)}\n\n", "#60A5FA")
        t_start = time.time()

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, bufsize=1, universal_newlines=True
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    if "[✔]" in line or "successfully" in line:
                        self.log_message(line, "#00FF66")
                    elif "⚠️" in line or "WARNING" in line or "Error" in line:
                        self.log_message(line, "#FBBF24")
                    else:
                        self.log_message(line)
                        
            rc = process.poll()
            if rc == 0:
                self.log_message(f"\n🎉 Parallel pipeline evaluation completed. Verifying multi-objective composite rankings...\n", "#00FF66")
                for target in target_ligands:
                    self.parse_and_print_summary(target)
                self.log_message(f"🎉 Total Execution Time for Run Session: {time.time() - t_start:.2f}s\n", "#00FF66")
            else:
                self.log_message(f"\n[PROCESS STATUS] Grid loop exited with non-zero error return code: {rc}\n", "#FBBF24")
                
        except Exception as e:
            self.log_message(f"\n[CRITICAL RUNTIME EXCEPTION] Driver loop communication failure: {str(e)}\n", "#FF3B30")
            
        finally:
            self.btn_run.config(state="normal")

if __name__ == "__main__":
    app = VADockerGUI()
    app.mainloop()