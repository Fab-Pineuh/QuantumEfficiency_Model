import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy.stats import qmc
import traceback
import threading
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")



PARAMS_BY_MODE = {
    "front": {
        "EgShift": (1.0, 0.95, 1.05),
        "Recomb": (1.0,0.8,1.0),
        "AZO": (160, 140, 180),
        "ZnO": (20, 10, 60),
        "CdS": (38, 30, 45),
        "CIGS": (500, 300, 2000),
        "dSCR": (450, 0, 2000),
        "Ln": (50, 0.01, 2000),
        "dDEAD": (0, 0, 500),
    },
    "rear": {
        "EgShift": (1.0, 0.95, 1.05),
        "Recomb": (1.0, 0.8, 1.0),
        "SLG": (1100000, 1090000, 1110000),
        "ITO": (100,90,115),
        "CIGS": (500, 300, 2000),
        "dSCR": (450, 0, 2000),
        "Ln": (50, 0.01, 2000),
        "dDEAD": (0, 0, 500),
    },
    "both": {
        "EgShift": (1.0, 0.95, 1.05),
        "Recomb": (1.0,0.8,1.0),
        "AZO": (160, 140, 180),
        "ZnO": (20, 10, 60),
        "CdS": (38, 30, 45),
        "SLG": (1100000, 1090000, 1110000),
        "ITO": (100,90,115),
        "CIGS": (500, 300, 2000),
        "dSCR": (450, 0, 2000),
        "Ln": (50, 0.01, 2000),
        "dDEAD": (0, 0, 500),
    }
}





# --- Load n,k data and interpolate to target wavelength ---
def load_and_interpolate_nk_csv(filepath, target_wavelengths):
    df = pd.read_csv(filepath, header=None, encoding='utf-8-sig')  # <- ici
    wl = df.iloc[:, 0].values
    n = df.iloc[:, 1].values
    k = df.iloc[:, 2].values
    interp_n = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
    interp_k = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")
    return interp_n(target_wavelengths), interp_k(target_wavelengths)

def compute_Topt(wavelength, nk_data, thicknesses):
    Topt = np.ones_like(wavelength, dtype=float)
    parasitic_abs=[]
    for (n, k), d in zip(nk_data, thicknesses):
        alpha = compute_alpha(k, wavelength)
        Topt *= np.exp(-alpha * d * 1e-7)
        parasitic_abs.append(np.exp(-alpha * d * 1e-7))
    return Topt, parasitic_abs

# --- Compute absorption coefficient alpha ---
def compute_alpha(k, wavelength):
    return 4 * np.pi * k / (wavelength * 1e-7)


# --- Compute IQE model ---
def compute_IQE_components_rear(wavelength, alpha_CIGS, Topt, d_SCR, L_n, d_CIGS, d_DEAD, recomb_factor):
    """
    Rear-side IQE model with dead zone:
    - d_DEAD is a region where light is absorbed but no carriers are collected.
    - Collection ends at d_coll = d_CIGS - d_DEAD
    """

   # Convert dimensions to cm
    w_cm = d_SCR * 1e-7       # SCR width
    d_cm = d_CIGS * 1e-7      # total CIGS thickness
    d_dead_cm = d_DEAD * 1e-7
    d_coll = d_cm - d_dead_cm
    
    # Clamp to ensure collection region is at least the SCR
    d_coll = max(d_coll, w_cm)


    # --- Drift collection (SCR) ---
    drift_term_exp = np.clip(alpha_CIGS * (d_cm - w_cm), 0, 700)
    exp_drift_rear = np.exp(-drift_term_exp)

    drift_term_exp2 = np.clip(alpha_CIGS * w_cm, 0, 700)
    drift_term = 1 - np.exp(-drift_term_exp2)

    IQE_drift = exp_drift_rear * drift_term

    # --- Diffusion collection (QNR) ---
    if L_n <= 0 or (d_coll - w_cm) <= 0:
        IQE_diff = np.zeros_like(alpha_CIGS)
    else:
        L_cm = L_n * 1e-7
        prefactor = (alpha_CIGS * L_cm) / (1 + alpha_CIGS * L_cm)
        term = np.clip((alpha_CIGS + 1 / L_cm) * (d_coll - w_cm), 0, 700)
        decay_term = np.exp(-term)
        IQE_diff = prefactor * (1 - decay_term)

    # Total IQE
    IQE_total = recomb_factor * Topt * (IQE_drift + IQE_diff)
    return IQE_total, recomb_factor * Topt * IQE_drift, recomb_factor * Topt * IQE_diff


def compute_IQE_components_front(wavelength, alpha_CIGS, Topt, d_SCR, L_n, d_CIGS, d_DEAD, recomb_factor):
    # Drift collection in the space-charge region (SCR)
    exp_drift = np.exp(-np.clip(alpha_CIGS * d_SCR * 1e-7, 0, 700))
    IQE_drift = 1 - exp_drift

    # Diffusion collection in the quasi-neutral region
    if L_n <= 0:
        IQE_diff = np.zeros_like(alpha_CIGS)  # No diffusion contribution
    else:
        delta = np.clip((d_CIGS - d_SCR) * 1e-7, 0, 1e4)
        term = np.clip(alpha_CIGS + 1 / (L_n * 1e-7), 0, 1e9)
        exp_diff = np.exp(-term * delta)
        IQE_diff = (L_n * 1e-7 * alpha_CIGS) / (1 + L_n * 1e-7 * alpha_CIGS) * exp_drift * (1 - exp_diff)

    return recomb_factor * Topt * (IQE_drift + IQE_diff), recomb_factor * Topt * IQE_drift, recomb_factor * Topt * IQE_diff


def compute_front_reflected_IQE(alpha_CIGS, Topt, d_CIGS, R_metal, wavelength, d_SCR, L_n, d_DEAD, recomb_factor):
    light = Topt * np.exp(-alpha_CIGS * d_CIGS * 1e-7)
    
    Reflected = light * R_metal
    Absorbed = light * (1 - R_metal)

    # Use front-side model for this second pass
    IQE_refl, drift_refl, diff_refl = compute_IQE_components_front(
        wavelength, alpha_CIGS, Reflected, d_SCR, L_n, d_CIGS, d_DEAD, 1
    )
    return IQE_refl, Reflected, Absorbed


# --- Initial parameters ---
def get_initial_params(mode):
    return {k: v[0] for k, v in PARAMS_BY_MODE[mode].items()}

def get_param_bounds(mode):
    return {k: (v[1], v[2]) for k, v in PARAMS_BY_MODE[mode].items()}



# --- Tkinter GUI ---
class IQEFitApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("IQE Fit GUI")
        self.eqe_file_front = tk.StringVar()
        self.eqe_file_rear = tk.StringVar()
        self.rfl_file_front = tk.StringVar()
        self.rfl_file_rear = tk.StringVar()
        self.create_widgets("front")
        self.update_mode()
        self.stored_values = get_initial_params("front")

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(1, weight=1)

        self.frame_right.grid_propagate(True)
            
    def update_entry(self, param, value):
        def update():
            self.entries[param].delete(0, tk.END)
            self.entries[param].insert(0, f"{value:.6f}" if param == "EgShift" else f"{value:.2f}")
        self.root.after(0, update)


    def load_data_files(self, illumination_mode):
        """Load EQE and reflectance data for the selected illumination mode."""
        if illumination_mode == "both":
            front = self.load_data_files("front")
            rear = self.load_data_files("rear")
            return (*front, *rear)
        if illumination_mode == "front":
            eqe_front = pd.read_csv(self.eqe_file_front.get(), header=None)
            rfl_front = pd.read_csv(self.rfl_file_front.get(), header=None)

            eqe_front.columns = ["wavelength", "EQE"]
            rfl_front.columns = ["wavelength", "Reflectance"]
            eqe_front = eqe_front[(eqe_front["wavelength"] >= 310) & (eqe_front["wavelength"] <= 830)]
            wl_front = eqe_front["wavelength"].values
            eqe_front_val = eqe_front["EQE"].values
            rfl_interp_f = interp1d(rfl_front["wavelength"], rfl_front["Reflectance"], bounds_error=False, fill_value="extrapolate")
            refl_front = rfl_interp_f(wl_front)
        elif illumination_mode == "rear":
            eqe_rear = pd.read_csv(self.eqe_file_rear.get(), header=None)
            rfl_rear = pd.read_csv(self.rfl_file_rear.get(), header=None)

            eqe_rear.columns = ["wavelength", "EQE"]
            rfl_rear.columns = ["wavelength", "Reflectance"]
            eqe_rear = eqe_rear[(eqe_rear["wavelength"] >= 310) & (eqe_rear["wavelength"] <= 830)]
            wl_rear = eqe_rear["wavelength"].values
            eqe_rear_val = eqe_rear["EQE"].values
            rfl_interp_r = interp1d(rfl_rear["wavelength"], rfl_rear["Reflectance"], bounds_error=False, fill_value="extrapolate")
            refl_rear = rfl_interp_r(wl_rear)

        if illumination_mode == "front":
            return wl_front, eqe_front_val, refl_front
        elif illumination_mode == "rear":
            return wl_rear, eqe_rear_val, refl_rear
        else:
            raise ValueError(f"Unknown illumination mode: {illumination_mode}")


    def browse_eqe(self, mode):
        """Load an EQE CSV file and store it for the given illumination side."""
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file:
            return

        if mode == "front":
            self.eqe_file_front.set(file)
        else:
            self.eqe_file_rear.set(file)

        self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get()))


    def browse_reflectance(self, mode):
        """Load a reflectance CSV file and store it for the given side."""
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file:
            return

        if mode == "front":
            self.rfl_file_front.set(file)
        else:
            self.rfl_file_rear.set(file)

        self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get()))
    
    def browse_metal_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.metal_file.set(file)
            if self.metal_file.get():
                self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get()))
    
    
    def build_param_inputs(self):
        # Clean up previous widgets
        for widget in self.frame_left.winfo_children():
            widget.destroy()
    
        self.entries = {}
        self.check_vars = {}
        self.stderr_labels = {}
    
        mode = self.illumination_mode.get()
        initials = get_initial_params(mode)
        bounds = get_param_bounds(mode)

        default_fixed = {
            "rear": ["SLG", "ITO", "CIGS", "Recomb", "dDEAD"],
            "front": ["AZO", "ZnO", "CdS", "CIGS", "dDEAD", "Recomb"],
            "both": ["AZO", "ZnO", "CdS", "SLG", "ITO", "CIGS", "dDEAD", "Recomb"]
        }
    
        for param in bounds:
            lb, ub = bounds [param]
            value = initials[param]
            group = ttk.LabelFrame(self.frame_left, text=param, padding=(10, 5))
            group.pack(fill="x", pady=5)
    
            # Row 1: Labels
            label_row = ttk.Frame(group)
            label_row.pack(fill="x")
            ttk.Label(label_row, text="Lower bound").grid(row=0, column=0, padx=5, sticky="w")
            ttk.Label(label_row, text=f"{param}").grid(row=0, column=1, padx=5)
            ttk.Label(label_row, text="Upper bound").grid(row=0, column=2, padx=5, sticky="e")
    
            # Row 2: Entry fields
            entry_row = ttk.Frame(group)
            entry_row.pack(fill="x")
            
            lb_entry = ttk.Entry(entry_row, width=8)
            lb_entry.insert(0, str(lb))
            lb_entry.grid(row=0, column=0, padx=5)
    
            main_entry = ttk.Entry(entry_row, width=8)
            main_entry.insert(0, str(value))
            main_entry.grid(row=0, column=1, padx=5)
    
            ub_entry = ttk.Entry(entry_row, width=8)
            ub_entry.insert(0, str(ub))
            ub_entry.grid(row=0, column=2, padx=5)
    
            self.entries[param] = main_entry
            self.entries[param + "_lb"] = lb_entry
            self.entries[param + "_ub"] = ub_entry

            self.stderr_labels[param] = ttk.Label(group, text="± ?")
            self.stderr_labels[param].pack(pady=2)
            
            fix_var = tk.BooleanVar(value=(param in default_fixed.get(mode, [])))
            fix_check = ttk.Checkbutton(group, text=f"Fix {param}", variable=fix_var)
            fix_check.pack(pady=3)
            self.check_vars[param] = fix_var



    def create_widgets(self, illumination_mode):
        self.entries = {}
        self.check_vars = {}


        frame_top = ttk.Frame(self.root)
        frame_top.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Label(frame_top, text="EQE File (FRONT):").grid(row=0, column=0)
        ttk.Entry(frame_top, textvariable=self.eqe_file_front, width=50).grid(row=0, column=1)
        ttk.Button(frame_top, text="Browse", command=lambda: self.browse_eqe("front")).grid(row=0, column=2)

        ttk.Label(frame_top, text="Reflectance File (FRONT):").grid(row=1, column=0)
        ttk.Entry(frame_top, textvariable=self.rfl_file_front, width=50).grid(row=1, column=1)
        ttk.Button(frame_top, text="Browse", command=lambda: self.browse_reflectance("front")).grid(row=1, column=2)
        
        ttk.Label(frame_top, text="EQE File (REAR):").grid(row=2, column=0)
        ttk.Entry(frame_top, textvariable=self.eqe_file_rear, width=50).grid(row=2, column=1)
        ttk.Button(frame_top, text="Browse", command=lambda: self.browse_eqe("rear")).grid(row=2, column=2)

        ttk.Label(frame_top, text="Reflectance File (REAR):").grid(row=3, column=0)
        ttk.Entry(frame_top, textvariable=self.rfl_file_rear, width=50).grid(row=3, column=1)
        ttk.Button(frame_top, text="Browse", command=lambda: self.browse_reflectance("rear")).grid(row=3, column=2)
        
        # Metal reflection checkbox and file input
        self.enable_metal_reflection = tk.BooleanVar(value=False)
        self.metal_file = tk.StringVar()
        
        metal_frame = ttk.Frame(frame_top)
        metal_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        
        illumination_select_frame= ttk.Frame(frame_top)
        illumination_select_frame.grid(row=0, rowspan=4, column=3, columnspan=3, sticky="w", pady=2)
        
        self.illumination_mode = tk.StringVar(value=illumination_mode)
        param_bounds = get_param_bounds(illumination_mode)
        
        ttk.Radiobutton(
            illumination_select_frame,
            text="Front Illumination",
            variable=self.illumination_mode,
            value="front",
            command=self.update_mode
        ).pack(anchor="w")
        ttk.Radiobutton(
            illumination_select_frame,
            text="Rear Illumination",
            variable=self.illumination_mode,
            value="rear",
            command=self.update_mode
        ).pack(anchor="w")
        ttk.Radiobutton(
            illumination_select_frame,
            text="Both front and rear illumination",
            variable=self.illumination_mode,
            value="both",
            command=self.update_mode
        ).pack(anchor="w")

        
        ttk.Checkbutton(metal_frame, text="Enable metal reflection", variable=self.enable_metal_reflection).grid(row=0, column=0, padx=5)
        ttk.Entry(metal_frame, textvariable=self.metal_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(metal_frame, text="Browse", command=self.browse_metal_file).grid(row=0, column=2, padx=5)
        


        # === Left panel split into scrollable top and static bottom ===
        # === Left panel split into scrollable top and static bottom ===
        left_panel = ttk.Frame(self.root)
        left_panel.grid(row=1, column=0, padx=10, pady=10, sticky="ns")
        
        left_panel.grid_rowconfigure(0, weight=1)
        
        
        self.frame_right = ttk.Frame(self.root)
        self.frame_right.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # --- Scrollable canvas for the matplotlib figure ---
        self.fig_canvas = tk.Canvas(self.frame_right)
        self.fig_canvas.pack(side="left", fill="both", expand=True)
        self.fig_scrollbar = ttk.Scrollbar(self.frame_right, orient="vertical", command=self.fig_canvas.yview)
        self.fig_scrollbar.pack(side="right", fill="y")
        self.fig_canvas.configure(yscrollcommand=self.fig_scrollbar.set)

        self.fig_frame = ttk.Frame(self.fig_canvas)
        self.fig_window = self.fig_canvas.create_window((0, 0), window=self.fig_frame, anchor="nw")

        def on_fig_frame_configure(event):
            self.fig_canvas.configure(scrollregion=self.fig_canvas.bbox("all"))

        def on_fig_canvas_configure(event):
            self.fig_canvas.itemconfig(self.fig_window, width=event.width)

        self.fig_frame.bind("<Configure>", on_fig_frame_configure)
        self.fig_canvas.bind("<Configure>", on_fig_canvas_configure)

        
        # ---- Scrollable frame for parameters ----
        scrollable_container = ttk.Frame(left_panel)
        scrollable_container.grid(row=0, column=0, sticky="nsew")
        
        scroll_canvas = tk.Canvas(scrollable_container, height=700)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(scrollable_container, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create the frame inside the canvas
        self.frame_left = ttk.Frame(scroll_canvas)
        
        # ❗ Store the window ID so we can resize it later
        window_id = scroll_canvas.create_window((0, 0), window=self.frame_left, anchor="nw")
        
        def on_canvas_configure(event):
            scroll_canvas.itemconfig(window_id, width=event.width)
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        
        scroll_canvas.bind("<Configure>", on_canvas_configure)
        
        # ---- Enable mousewheel scrolling over the parameter area ----
        def _on_mousewheel(canvas, event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.frame_left.bind(
            "<Enter>", lambda e: scroll_canvas.bind_all("<MouseWheel>", lambda ev: _on_mousewheel(scroll_canvas, ev))
        )
        self.frame_left.bind(
            "<Leave>", lambda e: scroll_canvas.unbind_all("<MouseWheel>")
        )

        def bind_canvas_scroll():
            if hasattr(self, "canvas_combined"):
                widget = self.canvas_combined.get_tk_widget()
                widget.bind(
                    "<Enter>",
                    lambda e: self.fig_canvas.bind_all("<MouseWheel>", lambda ev: _on_mousewheel(self.fig_canvas, ev)),
                )
                widget.bind(
                    "<Leave>", lambda e: self.fig_canvas.unbind_all("<MouseWheel>")
                )

        self._bind_canvas_scroll = bind_canvas_scroll
        
                

        
        # ---- Static frame for buttons (non-scrollable, fixed) ----
        self.buttons_frame = ttk.Frame(left_panel)
        self.buttons_frame.grid(row=1, column=0, sticky="ew", pady=10)
        
        self.button_fit = ttk.Button(self.buttons_frame, text="Calculate", command=self.start_fit_thread)
        self.button_fit.pack(pady=5)
        
        self.fast_fit_var = tk.BooleanVar(value=True)
        fast_fit_check = ttk.Checkbutton(self.buttons_frame, text="Fast Fit", variable=self.fast_fit_var)
        fast_fit_check.pack(pady=2)
        
        self.precise_fit_var = tk.BooleanVar(value=False)
        precise_fit_check = ttk.Checkbutton(self.buttons_frame, text="Precise Fit", variable=self.precise_fit_var)
        precise_fit_check.pack(pady=2)
        
        self.button_revert = ttk.Button(self.buttons_frame, text="Revert fit", command=self.revert_fit)
        self.button_revert.pack(pady=5)
        
        self.button_plot = ttk.Button(self.buttons_frame, text="Plot", command=lambda: self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get())))
        self.button_plot.pack(pady=5)
        
        self.button_save = ttk.Button(self.buttons_frame, text="Save All", command=lambda: self.save_all_results(self.illumination_mode.get()))
        self.button_save.pack(pady=5)
        
        self.r2_label = ttk.Label(self.buttons_frame, text="R² = ?")
        self.r2_label.pack()

    def update_mode(self):
        mode = self.illumination_mode.get()

        # Define subplot height ratios depending on the selected mode
        if mode in ("front", "rear"):
            # Main plot should be larger than residuals and collection plots
            height_ratios = [3, 1, 1]
        else:
            # Front and rear main plots with equal height
            height_ratios = [3, 3, 1, 1]

        # Destroy any existing widgets/figure to avoid overlaps
        if hasattr(self, "canvas_combined"):
            self.canvas_combined.get_tk_widget().destroy()

        if hasattr(self, "fig_combined"):
            plt.close(self.fig_combined)

        if hasattr(self, "fig_canvas"):
            self.fig_canvas.destroy()

        if hasattr(self, "frame_right"):
            self.frame_right.destroy()

        # Remove all remaining widgets so layout can be recreated cleanly
        for child in self.root.winfo_children():
            child.destroy()

        # Recreate the widget layout for the selected mode
        self.create_widgets(mode)

        # Make the figure taller so it is easier to inspect
        self.fig_combined = plt.Figure(figsize=(10, 12), dpi=100)

        if mode in ("front", "rear"):
            gs = GridSpec(
                3,
                1,
                figure=self.fig_combined,
                height_ratios=height_ratios,
                hspace=0.25,
            )
            self.ax_main = self.fig_combined.add_subplot(gs[0])
            self.ax_residuals = self.fig_combined.add_subplot(gs[1], sharex=self.ax_main)
            self.ax_collection = self.fig_combined.add_subplot(gs[2])
        else:
            gs = GridSpec(
                4,
                1,
                figure=self.fig_combined,
                height_ratios=height_ratios,
                hspace=0.25,
            )
            self.ax_main_front = self.fig_combined.add_subplot(gs[0])
            self.ax_main_rear = self.fig_combined.add_subplot(gs[1])
            self.ax_residuals = self.fig_combined.add_subplot(gs[2], sharex=self.ax_main_front)
            self.ax_collection = self.fig_combined.add_subplot(gs[3])

        self.canvas_combined = FigureCanvasTkAgg(self.fig_combined, master=self.fig_frame)
        self.canvas_combined.draw()
        self.canvas_combined.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if hasattr(self, "_bind_canvas_scroll"):
            self._bind_canvas_scroll()

        self.build_param_inputs()
        



    def start_fit_thread(self):
        self.root.after(0, lambda: self.button_fit.config(state='disabled'))
    
        def threaded_fit():
            self.run_fit(self.illumination_mode.get())
            self.root.after(0, lambda: self.button_fit.config(state='normal'))
    
        threading.Thread(target=threaded_fit, daemon=True).start()




    def revert_fit(self):
        for param, val in self.stored_values.items():
            def update_entry(p=param, v=val):
                self.entries[p].delete(0, tk.END)
                self.entries[p].insert(0, str(v))
            self.root.after(0, update_entry)


    def plot_current_params(self,illumination_mode):
        if illumination_mode == "front":
            required = [self.eqe_file_front.get(), self.rfl_file_front.get()]
        elif illumination_mode == "rear":
            required = [self.eqe_file_rear.get(), self.rfl_file_rear.get()]
        else:
            required = [self.eqe_file_front.get(), self.rfl_file_front.get(), self.eqe_file_rear.get(), self.rfl_file_rear.get()]
        if not all(required):
            return
    
        try:
            if illumination_mode in ["front", "rear"]:
                wavelength, EQE_meas, Reflectance = self.load_data_files(illumination_mode)
            else:
                wl_front, EQE_meas_front, R_front = self.load_data_files("front")
                wl_rear, EQE_meas_rear, R_rear = self.load_data_files("rear")
        except Exception as e:
            print("Erreur de chargement des fichiers :", e)
            return

        if illumination_mode in ["front", "rear"]:
            IQE_exp = EQE_meas / (1 - Reflectance)
        else:
            IQE_exp_front = EQE_meas_front / (1 - R_front)
            IQE_exp_rear = EQE_meas_rear / (1 - R_rear)

        try:
            if illumination_mode=="front":
                final_params = { param: float(self.entries[param].get()) for param in ["EgShift","dDEAD", "CIGS", "dSCR", "Ln", "Recomb","AZO", "ZnO", "CdS"] }
            elif illumination_mode=="rear":
                final_params = { param: float(self.entries[param].get()) for param in ["EgShift","dDEAD", "CIGS", "dSCR", "Ln", "Recomb", "SLG", "ITO"] }
            elif illumination_mode=="both":
                final_params = { param: float(self.entries[param].get()) for param in ["EgShift","dDEAD", "CIGS", "dSCR", "Ln", "Recomb","AZO", "ZnO", "CdS", "SLG", "ITO"] }
            
        except ValueError:
            self.ax_main.clear()
            self.ax_main.plot(wavelength, IQE_exp, 'ko', label='Exp. IQE')
            self.ax_main.set_xlabel('Wavelength (nm)')
            self.ax_main.set_ylabel('IQE')
            self.ax_main.set_xlim(310, 830)
            self.ax_main.set_xticks(np.arange(320, 840, 40))
            self.ax_main.set_ylim(0, 1)
            self.ax_main.grid(True)
            self.ax_main.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), fontsize=9)
            self.canvas_combined.draw()
            self.r2_label.config(text="R² = ?")
            return
        
        EgShift = final_params["EgShift"]

        if illumination_mode in ["front", "rear"]:
            wavelength_shifted = wavelength * EgShift
            n_cigs, k_cigs = load_and_interpolate_nk_csv('CIGSu.csv', wavelength_shifted)
            alpha_CIGS = compute_alpha(k_cigs, wavelength)
        else:
            wl_front_shift = wl_front * EgShift
            wl_rear_shift = wl_rear * EgShift
            n_cigs_f, k_cigs_f = load_and_interpolate_nk_csv('CIGSu.csv', wl_front_shift)
            n_cigs_r, k_cigs_r = load_and_interpolate_nk_csv('CIGSu.csv', wl_rear_shift)
            alpha_CIGS_front = compute_alpha(k_cigs_f, wl_front)
            alpha_CIGS_rear = compute_alpha(k_cigs_r, wl_rear)
        
        if illumination_mode=="front":
            n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wavelength)
            n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wavelength)
            n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wavelength)
            nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]
            
            Topt, parasitic_abs = compute_Topt(wavelength, nk_data, [final_params["AZO"], final_params["ZnO"], final_params["CdS"]])
            
            IQE_fit, drift_comp, diff_comp = compute_IQE_components_front(
                wavelength, alpha_CIGS, Topt, final_params["dSCR"], final_params["Ln"], final_params["CIGS"], final_params["dDEAD"], final_params["Recomb"]
            )
        elif illumination_mode=="rear":
            n_slg, k_slg = load_and_interpolate_nk_csv('SLG.csv', wavelength)
            n_ito, k_ito = load_and_interpolate_nk_csv('ITO.csv', wavelength)
            nk_data = [(n_slg, k_slg), (n_ito, k_ito)]
            Topt, parasitic_abs_rear = compute_Topt(
                wavelength,
                nk_data,
                [final_params["SLG"], final_params["ITO"]],
            )
            slg_abs = 1 - parasitic_abs_rear[0]
            ito_abs = parasitic_abs_rear[0] * (1 - parasitic_abs_rear[1])
            IQE_fit, drift_comp, diff_comp = compute_IQE_components_rear(
                wavelength,
                alpha_CIGS,
                Topt,
                final_params["dSCR"],
                final_params["Ln"],
                final_params["CIGS"],
                final_params["dDEAD"],
                final_params["Recomb"],
            )

        else:
            n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wl_front)
            n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wl_front)
            n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wl_front)
            nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]

            Topt_front, parasitic_abs_front = compute_Topt(
                wl_front,
                nk_data,
                [final_params["AZO"], final_params["ZnO"], final_params["CdS"]],
            )
            n_slg_r, k_slg_r = load_and_interpolate_nk_csv('SLG.csv', wl_rear)
            n_ito_r, k_ito_r = load_and_interpolate_nk_csv('ITO.csv', wl_rear)
            nk_data_rear = [(n_slg_r, k_slg_r), (n_ito_r, k_ito_r)]
            Topt_rear, parasitic_abs_rear = compute_Topt(wl_rear, nk_data_rear, [final_params["SLG"], final_params["ITO"]])
            slg_abs_rear = 1 - parasitic_abs_rear[0]
            ito_abs_rear = parasitic_abs_rear[0] * (1 - parasitic_abs_rear[1])

            IQE_fit_front, drift_comp_front, diff_comp_front = compute_IQE_components_front(
                wl_front,
                alpha_CIGS_front,
                Topt_front,
                final_params["dSCR"],
                final_params["Ln"],
                final_params["CIGS"],
                final_params["dDEAD"],
                final_params["Recomb"],
            )
            IQE_fit_rear, drift_comp_rear, diff_comp_rear = compute_IQE_components_rear(
                wl_rear,
                alpha_CIGS_rear,
                Topt_rear,
                final_params["dSCR"],
                final_params["Ln"],
                final_params["CIGS"],
                final_params["dDEAD"],
                final_params["Recomb"],
            )


        if illumination_mode != "both" and self.enable_metal_reflection.get() and self.metal_file.get():

            metal_df = pd.read_csv(self.metal_file.get(), header=None)
            metal_interp = interp1d(metal_df.iloc[:, 0], metal_df.iloc[:, 1], bounds_error=False, fill_value="extrapolate")
            R_metal = metal_interp(wavelength)
        
            IQE_refl, comeback_reflection, metal_absorbed = compute_front_reflected_IQE(
                alpha_CIGS,
                Topt,
                final_params["CIGS"],       # d_CIGS
                R_metal,
                wavelength,
                final_params["dSCR"],
                final_params["Ln"],
                final_params["dDEAD"],
                final_params["Recomb"]
            )
            IQE_fit_memory = IQE_fit
            IQE_fit += IQE_refl  # ← Add second-pass IQE
        
        # === Collection profile computation ===
        dSCR = final_params["dSCR"]
        Ln = final_params["Ln"]
        dCIGS = final_params["CIGS"]
        dDEAD = final_params["dDEAD"]
        
        depths = np.linspace(0, dCIGS, 500)
        collection = np.zeros_like(depths)
        
        x_coll_end = dCIGS - dDEAD
        x_Ln = dSCR + Ln
        
        if Ln <= 0:
            collection[:] = 0.0
        else:
            for i, x in enumerate(depths):
                if x <= dSCR:
                    collection[i] = 1.0
                elif x <= x_coll_end:
                    collection[i] = np.exp(-(x - dSCR) / Ln)
                else:
                    collection[i] = 0.0



    
        if illumination_mode != "both":
            r_squared = 1 - np.sum((IQE_fit - IQE_exp)**2) / np.sum((IQE_exp - np.mean(IQE_exp))**2)
    
        # Clear and reuse axes
        self.ax_main.clear()
        self.ax_residuals.clear()
        # --- Stackplot order: Reflected, Uncollected, Collected, Transmitted ---
        self.ax_main.clear()
        
        if illumination_mode == "both":
            self.ax_main_front.clear()
            self.ax_main_rear.clear()

            self.ax_main_front.stackplot(
                wl_front,
                IQE_fit_front,
                Topt_front * (1 - np.exp(-alpha_CIGS_front * dCIGS * 1e-7)) - IQE_fit_front,
                parasitic_abs_front[0] * parasitic_abs_front[1] * (1 - parasitic_abs_front[2]),
                parasitic_abs_front[0] * (1 - parasitic_abs_front[1]),
                1 - parasitic_abs_front[0],
                parasitic_abs_front[0]
                * parasitic_abs_front[1]
                * parasitic_abs_front[2]
                * np.exp(-alpha_CIGS_front * dCIGS * 1e-7),
                labels=["Collected", "Absorbed", "CdS", "ZnO", "AZO", "Transmitted"],
                colors=["#eff821", "#ffdd36", "#6fb802", "#9e0b0b", "#ff6600", "#00cbcc"],
            )
            self.ax_main_front.plot(wl_front, IQE_fit_front, "r-", linewidth=2, label="Fit (line)")
            self.ax_main_front.plot(
                wl_front,
                IQE_exp_front,
                "ko",
                markerfacecolor="none",
                markersize=8,
                markeredgewidth=1.2,
                label="Exp. IQE",
            )
            self.ax_main_front.plot(wl_front, drift_comp_front, "g--", label="SCR contribution")
            self.ax_main_front.plot(wl_front, diff_comp_front, "b--", label="Ln contribution")
            self.ax_main_front.set_ylabel("Quantum efficiency")
            self.ax_main_front.set_xlim(310, 830)
            self.ax_main_front.set_xticks(np.arange(320, 840, 40))
            self.ax_main_front.set_ylim(0, 1)
            self.ax_main_front.grid(True)
            self.ax_main_front.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

            self.ax_main_rear.stackplot(
                wl_rear,
                IQE_fit_rear,
                Topt_rear * (1 - np.exp(-alpha_CIGS_rear * dCIGS * 1e-7)) - IQE_fit_rear,
                Topt_rear * np.exp(-alpha_CIGS_rear * dCIGS * 1e-7),
                slg_abs_rear,
                ito_abs_rear,
                labels=["Collected", "Absorbed", "Transmitted", "SLG", "ITO"],
                colors=["#eff821", "#ffdd36", "#00cbcc", "#e7e68f", "#c0ffee"],
            )
            self.ax_main_rear.plot(wl_rear, IQE_fit_rear, "r-", linewidth=2, label="Fit (line)")
            self.ax_main_rear.plot(
                wl_rear,
                IQE_exp_rear,
                "ko",
                markerfacecolor="none",
                markersize=8,
                markeredgewidth=1.2,
                label="Exp. IQE",
            )
            self.ax_main_rear.plot(wl_rear, drift_comp_rear, "g--", label="SCR contribution")
            self.ax_main_rear.plot(wl_rear, diff_comp_rear, "b--", label="Ln contribution")
            self.ax_main_rear.set_ylabel("Quantum efficiency")
            self.ax_main_rear.set_xlabel("Wavelength (nm)")
            self.ax_main_rear.set_xlim(310, 830)
            self.ax_main_rear.set_xticks(np.arange(320, 840, 40))
            self.ax_main_rear.set_ylim(0, 1)
            self.ax_main_rear.grid(True)
            self.ax_main_rear.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
            
        elif self.enable_metal_reflection.get() and self.metal_file.get():
            if illumination_mode == "rear":
                self.ax_main.stackplot(
                    wavelength,
                    IQE_fit,
                    Topt * (1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)) - IQE_fit,
                    comeback_reflection * (1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)),
                    comeback_reflection * np.exp(-alpha_CIGS * dCIGS * 1e-7),
                    metal_absorbed,
                    slg_abs,
                    ito_abs,
                    labels=[
                        "Collected",
                        "Absorbed",
                        "Absorbed thanks to metal",
                        "Re-Transmitted",
                        "Absorbed (Metal)",
                        "SLG absorption",
                        "ITO absorption",
                    ],
                    colors=[
                        "#eff821",
                        "#ffdd36",
                        "#ff6600",
                        "#00cbcc",
                        "#c0c0c0",
                        "#c0ffee",
                        "#a0a0a0",
                    ],
                )
            else:  # front illumination
                self.ax_main.stackplot(
                    wavelength,
                    IQE_fit,
                    Topt * (1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)) - IQE_fit,
                    comeback_reflection * (1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)),
                    comeback_reflection * np.exp(-alpha_CIGS * dCIGS * 1e-7),
                    metal_absorbed,
                    1 - Topt,
                    labels=[
                        "Collected",
                        "Absorbed",
                        "Absorbed thanks to metal",
                        "Re-Transmitted",
                        "Absorbed (Metal)",
                        "Parasitic",
                    ],
                    colors=[
                        "#eff821",
                        "#ffdd36",
                        "#ff6600",
                        "#00cbcc",
                        "#c0c0c0",
                        "#c0ffee",
                    ],
                )
            
        elif illumination_mode=="rear":
            self.ax_main.stackplot(
                wavelength,
                IQE_fit,                                             # Collected
                Topt*(1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)) - IQE_fit,  # Absorbed but lost
                Topt*(np.exp(-alpha_CIGS * dCIGS * 1e-7)),                 # Transmitted
                slg_abs,
                ito_abs,
                labels=["Collected", "Absorbed", "Transmitted","SLG", "ITO"],
                colors=["#eff821", "#ffdd36", "#00cbcc", "#e7e68f", "#c0ffee"]
            )
        elif illumination_mode=="front":
            
            self.ax_main.stackplot(
                wavelength,
                IQE_fit,                                                      # Collected
                Topt*(1 - np.exp(-alpha_CIGS * dCIGS * 1e-7)) - IQE_fit,      # Absorbed but lost
                parasitic_abs[0]*parasitic_abs[1]*(1-parasitic_abs[2]),                                    # CdS
                parasitic_abs[0]*(1-parasitic_abs[1]),                   # ZnO
                1-parasitic_abs[0],  # AZO
                parasitic_abs[0] * parasitic_abs[1] * parasitic_abs[2] * np.exp(-alpha_CIGS * dCIGS * 1e-7),
                labels=["Collected", "Absorbed", "CdS","ZnO", "AZO", "Transmitted"],
                colors=["#eff821", "#ffdd36", "#6fb802", "#9e0b0b", "#ff6600", "#00cbcc"]
            )



        if illumination_mode != "both":
            self.ax_main.plot(wavelength, IQE_fit, "r-", linewidth=2, label="Fit (line)")
            self.ax_main.plot(
                wavelength,
                IQE_exp,
                "ko",
                markerfacecolor="none",
                markersize=8,
                markeredgewidth=1.2,
                label="Exp. IQE",
            )
            self.ax_main.plot(wavelength, drift_comp, "g--", label="SCR contribution")
            self.ax_main.plot(wavelength, diff_comp, "b--", label="Ln contribution")
            if self.enable_metal_reflection.get() and self.metal_file.get():
                self.ax_main.plot(wavelength, IQE_refl, "k--", label="Metal reflection")
            self.ax_main.set_xlabel("Wavelength (nm)")
            self.ax_main.set_ylabel("Quantum efficiency")
            self.ax_main.set_xlim(310, 830)
            self.ax_main.set_xticks(np.arange(320, 840, 40))
            self.ax_main.set_ylim(0, 1)
            self.ax_main.grid(True)
            self.ax_main.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
        # Residuals
        if illumination_mode != "both":
            residuals = IQE_fit - IQE_exp
            self.ax_residuals.plot(wavelength, residuals, "k-", linewidth=0.8)
            self.ax_residuals.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            self.ax_residuals.set_ylabel("Residual (Fit - Exp)")
            self.ax_residuals.set_xlabel("Wavelength (nm)")
            self.ax_residuals.set_xlim(310, 830)
            self.ax_residuals.set_xticks(np.arange(320, 840, 40))
            self.ax_residuals.grid(True)
        else:
            residuals_front = IQE_fit_front - IQE_exp_front
            residuals_rear = IQE_fit_rear - IQE_exp_rear
            self.ax_residuals.plot(wl_front, residuals_front, "r-", linewidth=0.8, label="Front")
            self.ax_residuals.plot(wl_rear, residuals_rear, "b-", linewidth=0.8, label="Rear")
            self.ax_residuals.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            self.ax_residuals.set_ylabel("Residual (Fit - Exp)")
            self.ax_residuals.set_xlabel("Wavelength (nm)")
            self.ax_residuals.set_xlim(310, 830)
            self.ax_residuals.set_xticks(np.arange(320, 840, 40))
            self.ax_residuals.grid(True)
            self.ax_residuals.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        



        self.ax_collection.clear()
        self.ax_collection.plot(depths, collection, 'r-')
        self.ax_collection.axvline(dSCR, color='blue', linestyle='--', linewidth=1, label=f"SCR={dSCR}")
        self.ax_collection.axvline(x_Ln, color='purple', linestyle='--', linewidth=1, label=f"Ln={Ln}")
        if dDEAD != 0:
            self.ax_collection.axvline(x_coll_end, color='green', linestyle='--', linewidth=1, label=f"Dead zone={dDEAD}")

        self.ax_collection.set_xlim(0, dCIGS)
        self.ax_collection.set_ylim(0, 1.05)
        self.ax_collection.set_xlabel('Depth in CIGS (nm)')
        self.ax_collection.set_ylabel('Collection efficiency')
        self.ax_collection.grid(True)
        self.ax_collection.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    
        # Final rendering
        self.fig_combined.subplots_adjust(right=0.80)
        self.canvas_combined.draw()
        if illumination_mode == "both":
            r_front = 1 - np.sum((IQE_fit_front - IQE_exp_front) ** 2) / np.sum((IQE_exp_front - np.mean(IQE_exp_front)) ** 2)
            r_rear = 1 - np.sum((IQE_fit_rear - IQE_exp_rear) ** 2) / np.sum((IQE_exp_rear - np.mean(IQE_exp_rear)) ** 2)
            self.r2_label.config(text=f"R² front={r_front:.4f} rear={r_rear:.4f}")
        else:
            self.r2_label.config(text=f"R² = {r_squared:.4f}")

    def save_all_results(self,illumination_mode):
        import csv

        if illumination_mode == "both":
            # Save front and rear data separately using the same logic
            self.save_all_results("front")
            self.save_all_results("rear")
            return
    
        title = "Save FRONT results" if illumination_mode=="front" else "Save REAR results"
        file_path = filedialog.asksaveasfilename(title=title, defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return  # user cancelled
    
        try:
            # === Prepare data ===
            wavelength, EQE_meas, Reflectance = self.load_data_files(illumination_mode)
            IQE_exp = EQE_meas / (1 - Reflectance)
            
            try:
                if illumination_mode=="front":
                    params = { param: float(self.entries[param].get()) for param in ["EgShift","dDEAD", "CIGS", "dSCR", "Ln", "Recomb","AZO", "ZnO", "CdS"] }
                elif illumination_mode=="rear":
                    params = { param: float(self.entries[param].get()) for param in ["EgShift","dDEAD", "CIGS", "dSCR", "Ln", "Recomb", "SLG", "ITO"] }
            except Exception as e:
                print("❌ Save error:", e)
                traceback.print_exc()
                
            bounds = {p: (float(self.entries[p + "_lb"].get()), float(self.entries[p + "_ub"].get())) for p in params}

            wavelength_shifted = wavelength * params["EgShift"]
    
            n_cigs, k_cigs = load_and_interpolate_nk_csv('CIGSu.csv', wavelength_shifted)
            alpha_CIGS = compute_alpha(k_cigs, wavelength)
    
            if illumination_mode=="front":
                n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wavelength)
                n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wavelength)
                n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wavelength)
                nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]

                Topt, parasitic_abs = compute_Topt(wavelength, nk_data, [params["AZO"], params["ZnO"], params["CdS"]])

                cds_abs = parasitic_abs[0] * parasitic_abs[1] * (1 - parasitic_abs[2])
                zno_abs = parasitic_abs[0] * (1 - parasitic_abs[1])
                azo_abs = 1 - parasitic_abs[0]

                IQE_fit, drift_comp, diff_comp = compute_IQE_components_front(
                    wavelength, alpha_CIGS, Topt, params["dSCR"], params["Ln"], params["CIGS"], params["dDEAD"], params["Recomb"]
                )
            elif illumination_mode=="rear":
                n_slg, k_slg = load_and_interpolate_nk_csv('SLG.csv', wavelength)
                n_ito, k_ito = load_and_interpolate_nk_csv('ITO.csv', wavelength)
                nk_data = [(n_slg, k_slg), (n_ito, k_ito)]
                Topt, parasitic_abs_rear = compute_Topt(
                    wavelength,
                    nk_data,
                    [params["SLG"], params["ITO"]],
                )
                slg_abs_arr = 1 - parasitic_abs_rear[0]
                ito_abs_arr = parasitic_abs_rear[0] * (1 - parasitic_abs_rear[1])
                IQE_fit, drift_comp, diff_comp = compute_IQE_components_rear(
                    wavelength,
                    alpha_CIGS,
                    Topt,
                    params["dSCR"],
                    params["Ln"],
                    params["CIGS"],
                    params["dDEAD"],
                    params["Recomb"],
                )
    
            # Metal reflection if enabled
            if self.enable_metal_reflection.get() and self.metal_file.get():
                metal_df = pd.read_csv(self.metal_file.get(), header=None)
                metal_interp = interp1d(metal_df.iloc[:, 0], metal_df.iloc[:, 1], bounds_error=False, fill_value="extrapolate")
                R_metal = metal_interp(wavelength)
    
                IQE_refl, comeback_reflection, metal_absorbed = compute_front_reflected_IQE(
                    alpha_CIGS, Topt, params["CIGS"], R_metal, wavelength, params["dSCR"], params["Ln"], params["dDEAD"], params["Recomb"]
                )
                IQE_fit_total = IQE_fit + IQE_refl
            else:
                IQE_refl = None
                comeback_reflection = None
                metal_absorbed = None
                IQE_fit_total = IQE_fit
    
            residual = IQE_fit_total - IQE_exp

            if IQE_refl is not None:
                absorbed_cigs = Topt * (1 - np.exp(-alpha_CIGS * params["CIGS"] * 1e-7)) - IQE_fit_total
                absorbed_from_metal = comeback_reflection * (1 - np.exp(-alpha_CIGS * params["CIGS"] * 1e-7))
                transmitted = comeback_reflection * np.exp(-alpha_CIGS * params["CIGS"] * 1e-7)
                slg_abs = slg_abs_arr
                ito_abs = ito_abs_arr
            else:
                absorbed_cigs = Topt * (1 - np.exp(-alpha_CIGS * params["CIGS"] * 1e-7)) - IQE_fit
                transmitted = Topt * np.exp(-alpha_CIGS * params["CIGS"] * 1e-7)
                if illumination_mode == "rear":
                    slg_abs = slg_abs_arr
                    ito_abs = ito_abs_arr
                else:
                    slg_abs = ito_abs = None
    
            # Collection profile
            depths = np.linspace(0, params["CIGS"], 500)
            collection = np.zeros_like(depths)
            x_coll_end = params["CIGS"] - params["dDEAD"]
            for i, x in enumerate(depths):
                if x <= params["dSCR"]:
                    collection[i] = 1.0
                elif x <= x_coll_end:
                    collection[i] = np.exp(-(x - params["dSCR"]) / params["Ln"])
                else:
                    collection[i] = 0.0
    
            # === Write to CSV ===
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
    
                # Parameters section
                writer.writerow(["Illumination Mode", illumination_mode])
                writer.writerow([])

                writer.writerow(["Parameter", "Lower Bound", "Value", "StdDev", "Upper Bound"])
                for p in params:
                    lb, ub = bounds[p]
                    std = getattr(self, "param_errors", {}).get(p, "")
                    writer.writerow([p, lb, params[p], std, ub])
                writer.writerow([])
    
                # Main curves
                header = ["Wavelength", "Experimental_IQE", "IQE_fit", "IQE_fit_SCR", "IQE_fit_Ln"]
                if illumination_mode == "rear":
                    header.append("Topt")

                if IQE_refl is not None:
                    header += [
                        "IQE_fit_metal_reflection",
                        "Absorbed_CIGS",
                        "Absorbed_from_metal",
                    ]
                    header += ["ReTransmitted", "Absorbed_Metal"]
                    if illumination_mode == "rear":
                        header += ["SLG_absorption", "ITO_absorption"]
                else:

                    header += ["Absorbed_CIGS"]

                    if illumination_mode == "front":
                        header += [
                            "CdS_absorption",
                            "ZnO_absorption",
                            "AZO_absorption",
                        ]
                    header += ["Transmitted"]
                    if illumination_mode == "rear":
                        header += ["SLG_absorption", "ITO_absorption"]
                header += ["Residual"]

                writer.writerow(header)

    
                for i in range(len(wavelength)):
                    row = [
                        wavelength[i],
                        IQE_exp[i],
                        IQE_fit_total[i],
                        drift_comp[i],
                        diff_comp[i]
                    ]
                    if illumination_mode == "rear":
                        row.append(Topt[i])

                    if IQE_refl is not None:
                        row += [
                            IQE_refl[i],
                            absorbed_cigs[i],
                        ]
                        row += [absorbed_from_metal[i]]
                        row += [transmitted[i], metal_absorbed[i]]
                        if illumination_mode == "rear":
                            row += [slg_abs[i], ito_abs[i]]
                    else:
                        row += [absorbed_cigs[i]]
                        if illumination_mode == "front":
                            row += [cds_abs[i], zno_abs[i], azo_abs[i]]
                        row += [transmitted[i]]
                        if illumination_mode == "rear":
                            row += [slg_abs[i], ito_abs[i]]
                    row.append(residual[i])
                    writer.writerow(row)
    
                # Collection profile
                writer.writerow(["Depth_in_CIGS", "Collection_Efficiency"])
                for i in range(len(depths)):
                    writer.writerow([depths[i], collection[i]])
    
            print(f"✅ Data saved to {file_path}")

            # --- Save figures as PNG files ---
            folder = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            self.canvas_combined.draw()
            png_path = os.path.join(folder, f"{base}.png")
            self.fig_combined.savefig(png_path, bbox_inches="tight")
            print("✅ Figure saved as PNG")
        except Exception as e:
            print("❌ Save error:", e)
            traceback.print_exc()
            
    
    def update_after_fit(self, result, updated_values, errors=None):
        for param, value in updated_values.items():
            self.entries[param].delete(0, tk.END)
            if param == "EgShift":
                self.entries[param].insert(0, f"{value:.6f}")
            else:
                self.entries[param].insert(0, f"{value:.2f}")
            if errors and param in errors and param in self.stderr_labels:
                self.stderr_labels[param].config(text=f"±{errors[param]:.2f}")

        if errors:
            for p in self.stderr_labels:
                if p not in updated_values:
                    self.stderr_labels[p].config(text="± ?")

        self.plot_current_params(self.illumination_mode.get())

    
    

    def run_fit(self, illumination_mode):
        
        if illumination_mode == "front":
            parameters_list = ["dDEAD", "dSCR", "Ln", "CIGS", "EgShift", "Recomb", "AZO", "ZnO", "CdS"]
            required = [self.eqe_file_front.get(), self.rfl_file_front.get()]
            wl, EQE_meas, Reflectance = self.load_data_files("front")
            IQE_exp = EQE_meas / (1 - Reflectance)
        elif illumination_mode == "rear":
            parameters_list = ["dDEAD", "dSCR", "Ln", "CIGS", "EgShift", "Recomb", "SLG", "ITO"]
            required = [self.eqe_file_rear.get(), self.rfl_file_rear.get()]
            wl, EQE_meas, Reflectance = self.load_data_files("rear")
            IQE_exp = EQE_meas / (1 - Reflectance)
        else:
            parameters_list = ["dDEAD", "dSCR", "Ln", "CIGS", "EgShift", "Recomb", "AZO", "ZnO", "CdS", "SLG", "ITO"]
            required = [
                self.eqe_file_front.get(),
                self.rfl_file_front.get(),
                self.eqe_file_rear.get(),
                self.rfl_file_rear.get(),
            ]
            wl_front, EQE_front, R_front = self.load_data_files("front")
            wl_rear, EQE_rear, R_rear = self.load_data_files("rear")
            IQE_exp_front = EQE_front / (1 - R_front)
            IQE_exp_rear = EQE_rear / (1 - R_rear)

        if not all(required):
            return

        if illumination_mode in ["front", "rear"]:
            wavelength = wl
        else:
            wavelength = wl_front  # used for initial nk interpolation


        params = {param: float(self.entries[param].get()) for param in self.entries}
        self.stored_values = params.copy()
    
        x0 = []
        bounds = []
        fixed_flags = []
        
        for param in parameters_list:
            if self.check_vars[param].get():
                fixed_flags.append(True)
            else:
                fixed_flags.append(False)
                x0.append(params[param])
                lb = float(self.entries[param + "_lb"].get())
                ub = float(self.entries[param + "_ub"].get())
                bounds.append((lb, ub))


        def model_objective(x, illumination_mode):
            try:
                idx = 0
                p = params.copy()
                for i, param in enumerate(parameters_list):
                    if not fixed_flags[i]:
                        p[param] = x[idx]
                        idx += 1
                        
                if illumination_mode in ["front", "rear"]:
                    wl_use = wl if illumination_mode in ["front", "rear"] else wl_front
                    wl_shifted = wl_use * p["EgShift"]
                    n_cigs, k_cigs = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted)
                    alpha_CIGS = compute_alpha(k_cigs, wl_use)
                else:
                    wl_shifted_f = wl_front * p["EgShift"]
                    wl_shifted_r = wl_rear * p["EgShift"]
                    n_cigs_f, k_cigs_f = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted_f)
                    n_cigs_r, k_cigs_r = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted_r)
                    alpha_CIGS_front = compute_alpha(k_cigs_f, wl_front)
                    alpha_CIGS_rear = compute_alpha(k_cigs_r, wl_rear)
        

                if illumination_mode == "front":
                    n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wl)
                    n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wl)
                    n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wl)
                    nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]

                    Topt, _ = compute_Topt(wl, nk_data, [p["AZO"], p["ZnO"], p["CdS"]])

                    IQE_fit, _, _ = compute_IQE_components_front(
                        wl, alpha_CIGS, Topt, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    return np.sum((IQE_fit - IQE_exp) ** 2) / np.sum((IQE_exp - np.mean(IQE_exp)) ** 2)

                elif illumination_mode == "rear":
                    n_slg, k_slg = load_and_interpolate_nk_csv('SLG.csv', wl)
                    n_ito, k_ito = load_and_interpolate_nk_csv('ITO.csv', wl)
                    nk_data = [(n_slg, k_slg), (n_ito, k_ito)]
                    Topt, _ = compute_Topt(wl, nk_data, [p["SLG"], p["ITO"]])
                    IQE_fit, _, _ = compute_IQE_components_rear(
                        wl, alpha_CIGS, Topt, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    return np.sum((IQE_fit - IQE_exp) ** 2) / np.sum((IQE_exp - np.mean(IQE_exp)) ** 2)

                else:
                    n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wl_front)
                    n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wl_front)
                    n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wl_front)
                    nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]

                    Topt_front, _ = compute_Topt(wl_front, nk_data, [p["AZO"], p["ZnO"], p["CdS"]])
                    IQE_fit_front, _, _ = compute_IQE_components_front(
                        wl_front, alpha_CIGS_front, Topt_front, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    
                    n_slg_r, k_slg_r = load_and_interpolate_nk_csv('SLG.csv', wl_rear)
                    n_ito_r, k_ito_r = load_and_interpolate_nk_csv('ITO.csv', wl_rear)
                    nk_data_rear = [(n_slg_r, k_slg_r), (n_ito_r, k_ito_r)]
                    Topt_rear, _ = compute_Topt(wl_rear, nk_data_rear, [p["SLG"], p["ITO"]])
                    
                    IQE_fit_rear, _, _ = compute_IQE_components_rear(
                        wl_rear, alpha_CIGS_rear, Topt_rear, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )

                    err_front = np.sum((IQE_fit_front - IQE_exp_front) ** 2) / np.sum((IQE_exp_front - np.mean(IQE_exp_front)) ** 2)
                    err_rear = np.sum((IQE_fit_rear - IQE_exp_rear) ** 2) / np.sum((IQE_exp_rear - np.mean(IQE_exp_rear)) ** 2)
                    return err_front + err_rear
            
            
            
            
            
            except Exception as e:
                print("Objective error:", e)
                return np.inf

        def model_residuals(x, illumination_mode):
            try:
                idx = 0
                p = params.copy()
                for i, param in enumerate(parameters_list):
                    if not fixed_flags[i]:
                        p[param] = x[idx]
                        idx += 1

                if illumination_mode in ["front", "rear"]:
                    wl_use = wl if illumination_mode in ["front", "rear"] else wl_front
                    wl_shifted = wl_use * p["EgShift"]
                    n_cigs, k_cigs = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted)
                    alpha_CIGS = compute_alpha(k_cigs, wl_use)
                else:
                    wl_shifted_f = wl_front * p["EgShift"]
                    wl_shifted_r = wl_rear * p["EgShift"]
                    n_cigs_f, k_cigs_f = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted_f)
                    n_cigs_r, k_cigs_r = load_and_interpolate_nk_csv('CIGSu.csv', wl_shifted_r)
                    alpha_CIGS_front = compute_alpha(k_cigs_f, wl_front)
                    alpha_CIGS_rear = compute_alpha(k_cigs_r, wl_rear)

                if illumination_mode == "front":
                    n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wl)
                    n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wl)
                    n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wl)
                    nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]
                    Topt, _ = compute_Topt(wl, nk_data, [p["AZO"], p["ZnO"], p["CdS"]])
                    IQE_fit, _, _ = compute_IQE_components_front(
                        wl, alpha_CIGS, Topt, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    return IQE_fit - IQE_exp

                elif illumination_mode == "rear":
                    n_slg, k_slg = load_and_interpolate_nk_csv('SLG.csv', wl)
                    n_ito, k_ito = load_and_interpolate_nk_csv('ITO.csv', wl)
                    nk_data = [(n_slg, k_slg), (n_ito, k_ito)]
                    Topt, _ = compute_Topt(wl, nk_data, [p["SLG"], p["ITO"]])
                    IQE_fit, _, _ = compute_IQE_components_rear(
                        wl, alpha_CIGS, Topt, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    return IQE_fit - IQE_exp

                else:
                    n_azo, k_azo = load_and_interpolate_nk_csv('AZO.csv', wl_front)
                    n_zno, k_zno = load_and_interpolate_nk_csv('iZnO.csv', wl_front)
                    n_cds, k_cds = load_and_interpolate_nk_csv('CdS.csv', wl_front)
                    nk_data = [(n_azo, k_azo), (n_zno, k_zno), (n_cds, k_cds)]
                    Topt_front, _ = compute_Topt(wl_front, nk_data, [p["AZO"], p["ZnO"], p["CdS"]])
                    IQE_fit_front, _, _ = compute_IQE_components_front(
                        wl_front, alpha_CIGS_front, Topt_front, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    n_slg_r, k_slg_r = load_and_interpolate_nk_csv('SLG.csv', wl_rear)
                    n_ito_r, k_ito_r = load_and_interpolate_nk_csv('ITO.csv', wl_rear)
                    nk_data_rear = [(n_slg_r, k_slg_r), (n_ito_r, k_ito_r)]
                    Topt_rear, _ = compute_Topt(wl_rear, nk_data_rear, [p["SLG"], p["ITO"]])
                    IQE_fit_rear, _, _ = compute_IQE_components_rear(
                        wl_rear, alpha_CIGS_rear, Topt_rear, p["dSCR"], p["Ln"], p["CIGS"], p["dDEAD"], p["Recomb"]
                    )
                    return np.concatenate([IQE_fit_front - IQE_exp_front, IQE_fit_rear - IQE_exp_rear])

            except Exception:
                return np.ones_like(IQE_exp) * np.inf

        def constraint_sum_thickness(x):
            idx_map = {"dSCR": None, "Ln": None, "dDEAD": None, "CIGS": None}
            idx = 0
            for i, param in enumerate(parameters_list):
                if not fixed_flags[i]:
                    if param in idx_map:
                        idx_map[param] = idx
                    idx += 1    
        
            dSCR = x[idx_map["dSCR"]] if idx_map["dSCR"] is not None else params["dSCR"]
            Ln = x[idx_map["Ln"]] if idx_map["Ln"] is not None else params["Ln"]
            dDEAD = x[idx_map["dDEAD"]] if idx_map["dDEAD"] is not None else params["dDEAD"]
            dCIGS = x[idx_map["CIGS"]] if idx_map["CIGS"] is not None else params["CIGS"]
        
            return dCIGS - dSCR - Ln - dDEAD

        if all(self.check_vars[p].get() for p in ["dSCR", "Ln", "dDEAD", "CIGS"]):
            nonlinear_constraint = []
        else:
            nonlinear_constraint = [NonlinearConstraint(constraint_sum_thickness, 0, np.inf)]
        
        for param in parameters_list:
            if not self.check_vars[param].get():
                lb = float(self.entries[param + "_lb"].get())
                ub = float(self.entries[param + "_ub"].get())
                if lb >= ub:
                    tk.messagebox.showerror("Invalid Bounds", f"Lower bound must be < upper bound for {param}")
                    return
                
        best_result = None
        best_score = np.inf      
        
        
        # Number of samples
        n_samples = 16
        
        # Mean = current values from entries
        means = np.array([
            float(self.entries[param].get())
            for i, param in enumerate(parameters_list)
            if not fixed_flags[i]
        ])
        
        # Bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        
        refine_radius = 0.6  # initial exploration range (fraction of full bounds)
        
        current_mean = means.copy()
        
        # Initial best score: evaluate the current GUI parameters as a baseline
        initial_guess = []
        for i, param in enumerate(parameters_list):
            if not fixed_flags[i]:
                print(param)
                initial_guess.append(params[param])
        print("\n⚙️ Evaluating initial GUI parameters...")
        print("Initial guess:", initial_guess)
        res_init = minimize(
            lambda x: model_objective(x, self.illumination_mode.get()),
            initial_guess,
            method="trust-constr",
            bounds=bounds,
            constraints=nonlinear_constraint,
            options={"disp": False, "maxiter": 1}
        )
        
        if np.isfinite(res_init.fun):
            best_score = res_init.fun
            best_result = res_init
            current_mean = np.array(initial_guess)
            print("✅ Initial fit score from GUI values:", best_score)
        else:
            print("⚠️ Failed to evaluate initial GUI parameters.")

        no_improvement_counter = 0  # ← counts stages without any improvement
        
        # If Fast Fit is enabled, use a single stage and small sample set
        if self.precise_fit_var.get() and self.fast_fit_var.get():
            sample_sizes = [32]
            max_stages = 1
        elif self.fast_fit_var.get():
            sample_sizes = [16]   # One round only
            max_stages = 1
            print("⚡ Fast Fit mode enabled")
        else:
            sample_sizes = [32, 16, 8, 4]
            max_stages = 8

        
        stage = 0
        while stage < max_stages:
            if self.precise_fit_var.get() and self.fast_fit_var.get():
                refine_radius = 0.05
                radius = refine_radius * (0.6 ** stage)
            else:
                radius = refine_radius * (0.6 ** stage)
            print(f"\n🔁 Stage {stage+1} (radius={radius:.4f})")
            
            stage_improved = False  # ← flag to track if any improvement in this stage
        
            for n_samples in sample_sizes:
                print(f"   🔹 n_samples = {n_samples}")
                maxiter = int(1536 / n_samples)
        
                # Create Sobol samples around current_mean
                scaled_center = (current_mean - lower_bounds) / (upper_bounds - lower_bounds)
                sampler = qmc.Sobol(d=len(means), scramble=True, seed=42)
                sobol_points = sampler.random(n=n_samples)
                scaled_samples = scaled_center + (sobol_points - 0.5) * radius * 2
                scaled_samples = np.clip(scaled_samples, 0, 1)
                sobol_samples = lower_bounds + scaled_samples * (upper_bounds - lower_bounds)
        
                for x0_perturbed in sobol_samples:
                    res = minimize(
                        lambda x: model_objective(x, self.illumination_mode.get()),
                        x0_perturbed,
                        method="trust-constr",
                        bounds=bounds,
                        constraints=nonlinear_constraint,
                        options={
                            "verbose": 0,
                            "xtol": 1e-9,
                            "gtol": 1e-6,
                            "barrier_tol": 1e-8,
                            "maxiter": maxiter
                        }
                    )
                    if res.success and res.fun < best_score:
                        best_result = res
                        best_score = res.fun
                        current_mean = res.x
                        stage_improved = True
                        
                        # --- Update GUI entries ---
                        idx = 0
                        for i, param in enumerate(parameters_list):
                            if not fixed_flags[i]:
                                self.update_entry(param, res.x[idx])
                                idx += 1
                        
                        # --- Plot updated fit ---
                        self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get()))

        
                        # ✅ Log best values
                        idx = 0
                        print_values = {}
                        for i, param in enumerate(parameters_list):
                            if fixed_flags[i]:
                                print_values[param] = params[param]
                            else:
                                print_values[param] = res.x[idx]
                                idx += 1
        
                        formatted_values = [
                            f"{k} = {v:.6f}" if k == "EgShift" else f"{k} = {v:.2f}"
                            for k, v in print_values.items()
                        ]
                        print("   ✅ New best: " + ", ".join(formatted_values))
                        

            
            # Check if stage brought improvement
            if stage_improved:
                print("   🎯 Improvement found in stage")
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                print(f"   ⚠️ No improvement in this stage (counter={no_improvement_counter})")
        
            if no_improvement_counter >= 3:
                print(f"⏹️ No improvement for {no_improvement_counter} consecutive stages. Stopping optimization.")
                break
        
            stage += 1  # Go to next radius stage
        
        if best_result:
            result = best_result
            print("✅ Best fit selected.")
        else:
            print("❌ All optimization attempts failed.")
            return
        
        if not result.success:
            print("Fit failed:", result.message)
        
        idx = 0
        updated_values = {}
        for i, param in enumerate(parameters_list):
            if not fixed_flags[i]:
                updated_values[param] = result.x[idx]
                idx += 1

        # --- Estimate parameter uncertainties ---
        def residuals_func(x):
            return model_residuals(x, self.illumination_mode.get())

        try:
            from scipy.optimize._numdiff import approx_derivative

            J = approx_derivative(residuals_func, result.x)
            resid = residuals_func(result.x)
            dof = max(len(resid) - len(result.x), 1)
            sigma2 = np.sum(resid ** 2) / dof
            cov = sigma2 * np.linalg.pinv(J.T @ J)
            stds = np.sqrt(np.diag(cov))
            errors = {}
            idx = 0
            for i, param in enumerate(parameters_list):
                if not fixed_flags[i]:
                    errors[param] = stds[idx]
                    idx += 1
            self.param_errors = errors
        except Exception as e:
            print("⚠️ Could not compute parameter uncertainties:", e)
            self.param_errors = {}

        self.root.after(0, lambda: self.plot_current_params(self.illumination_mode.get()))
        self.root.after(0, lambda: self.update_after_fit(result, updated_values, self.param_errors))


# --- Run app ---
if __name__ == "__main__":
    root = tk.Tk()
    app = IQEFitApp(root)

    root.geometry("1400x900")
    root.minsize(1024, 700)

    root.mainloop()
