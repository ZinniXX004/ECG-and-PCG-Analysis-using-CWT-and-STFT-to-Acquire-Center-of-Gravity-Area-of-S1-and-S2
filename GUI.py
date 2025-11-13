# GUI.py (CWT+STFT integrated; UI-only extensions for STFT tab)
import os
import json
import traceback
import numpy as np
from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import cm

# New imports for STFT and 3D plotting
try:
    from STFT_PCG import compute_stft
except Exception:
    # If STFT_PCG not found, we will surface error when STFT is selected
    compute_stft = None
from mpl_toolkits.mplot3d import Axes3D  # for 3D surface plotting (import for side-effects)

# Local modules - unchanged algorithmic logic
from utils_io import try_load_record
from Pan_Tompkins import detect_r_peaks_with_fallback, plot_pt_pipeline
from Segmentation_ECG_to_PCG import segment_one_cycle
from CWT_PCG import compute_cwt
from Threshold_Plot_CoG import threshold_mask, compute_cog

try:
    from scipy import ndimage as sp_ndimage
except Exception:
    sp_ndimage = None

# Defaults (unchanged)
DEFAULT_FS = 2000
DEFAULT_CWT_FMIN = 5.0
DEFAULT_CWT_FMAX = 200.0
DEFAULT_CWT_NFREQS = 120
DEFAULT_CWT_COLCOUNT = 300
DEFAULT_PASCAL_A0 = 0.0019
DEFAULT_PASCAL_A_STEP = 0.00031

# UI theme + Matplotlib dark colors
QT_STYLE = """
QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #071224, stop:1 #0b3b70); color: #E6F0FA; font-family: "Segoe UI", Roboto, Arial; font-size: 11px; }
QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #09547a, stop:1 #0b73b3); color: #F3FBFF; border-radius:6px; padding:6px 10px; }
QPushButton:disabled { background:#1b2f44; color:#7f97b0; }
QLineEdit { background: rgba(255,255,255,0.04); color:#eaf6ff; border-radius:4px; padding:4px; }
QLabel { color: #dbeeff; }
QSlider::groove:horizontal { background: rgba(255,255,255,0.06); height:8px; border-radius:4px; }
QSlider::handle:horizontal { background: #0b73b3; width: 14px; border-radius:7px; margin:-3px 0; }
QComboBox { background: rgba(255,255,255,0.04); color:#eaf6ff; border-radius:4px; padding:4px; }
QSpinBox, QDoubleSpinBox { background: rgba(255,255,255,0.04); color:#eaf6ff; border-radius:4px; padding:2px 4px; }
QCheckBox { color:#eaf6ff; }
"""

FIG_FG = "#E6F0FA"
FIG_BG = "#071224"
AX_BG = "#071224"
GRID_COLOR = "#1B4D9E"

class PCGAnalyzerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCG/ECG Analyzer (patched UI + STFT)")
        self.resize(1250, 920)
        self.setStyleSheet(QT_STYLE)

        # state (unchanged)
        self.record_p_signal = None
        self.fs = DEFAULT_FS
        self.sig_names = []
        self.ecg_idx = None
        self.pcg_idx = None
        self.r_peaks = np.array([], dtype=int)

        # caches
        self.current_segment = None
        self.segment_bounds = (0, 0)
        self.current_scalogram = None
        self.current_freqs = None
        self.current_times = None
        self.current_cwt_method = None
        self.current_masks = {}
        self.current_cogs = {}

        # plot handles
        self.fig_pt_raw = None; self.canvas_pt_raw = None; self.ax_pt_ecg_raw = None
        self.fig_pt_proc = None; self.canvas_pt_proc = None; self.ax_pt_ecg_proc = None
        self.fig_pt_pcg = None; self.canvas_pt_pcg = None; self.ax_pt_pcg = None

        self.fig_seg_ecg = None; self.canvas_seg_ecg = None; self.ax_seg_ecg = None
        self.fig_seg_pcg = None; self.canvas_seg_pcg = None; self.ax_seg_pcg = None

        self.fig_cwt = None; self.canvas_cwt = None; self.ax_cwt = None
        self.fig_thr = None; self.canvas_thr = None; self.ax_thr = None

        # STFT figs
        self.fig_stft_2d = None; self.canvas_stft_2d = None; self.ax_stft_2d = None
        self.fig_stft_3d = None; self.canvas_stft_3d = None; self.ax_stft_3d = None

        self.cwt_cb = None; self.cwt_cax = None
        self.thr_cb = None; self.thr_cax = None
        self.cwt_im = None; self.thr_im = None
        self.last_thr_contours = []
        self.last_thr_scat = []

        self._build_ui()
        self._connect_signals()

    # ---------------- UI build ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_v = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        top_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load (.hea/.dat)")
        self.swap_btn = QtWidgets.QPushButton("Swap PCG/ECG")
        self.clear_btn = QtWidgets.QPushButton("CLEAR")
        self.save_btn = QtWidgets.QPushButton("Save result (PNG + JSON)")
        self.quit_btn = QtWidgets.QPushButton("QUIT")
        self.swap_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        top_row.addWidget(self.load_btn)
        top_row.addWidget(self.swap_btn)
        top_row.addWidget(self.clear_btn)
        top_row.addStretch()
        top_row.addWidget(self.save_btn)
        top_row.addWidget(self.quit_btn)
        main_v.addLayout(top_row)

        self.record_label = QtWidgets.QLabel("No record loaded")
        main_v.addWidget(self.record_label)

        self.tabs = QtWidgets.QTabWidget()
        main_v.addWidget(self.tabs, stretch=1)

        # PT tab with separate scrollable canvases
        self.tab_pt = QtWidgets.QWidget()
        t1 = QtWidgets.QVBoxLayout(self.tab_pt)
        pt_ctrl = QtWidgets.QHBoxLayout()
        self.pt_detect_btn = QtWidgets.QPushButton("Run Pan-Tompkins")
        self.pt_detect_btn.setEnabled(False)
        self.show_pt_pipeline_btn = QtWidgets.QPushButton("Show PT pipeline")
        self.show_pt_pipeline_btn.setEnabled(False)
        pt_ctrl.addWidget(self.pt_detect_btn)
        pt_ctrl.addWidget(self.show_pt_pipeline_btn)
        pt_ctrl.addStretch()
        t1.addLayout(pt_ctrl)

        self._create_pt_plot("raw", t1, height=260)
        self._create_pt_plot("proc", t1, height=220)
        self._create_pt_plot("pcg", t1, height=260)
        self.tabs.addTab(self.tab_pt, "Pan-Tompkins")

        # Segmentation tab (two independent canvases)
        self.tab_seg = QtWidgets.QWidget()
        t2 = QtWidgets.QVBoxLayout(self.tab_seg)
        seg_ctrl = QtWidgets.QHBoxLayout()
        seg_ctrl.addWidget(QtWidgets.QLabel("Beat index:"))
        self.seg_beat_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seg_beat_slider.setEnabled(False)
        self.seg_beat_label = QtWidgets.QLabel("0")
        seg_ctrl.addWidget(self.seg_beat_slider)
        seg_ctrl.addWidget(self.seg_beat_label)
        seg_ctrl.addStretch()
        t2.addLayout(seg_ctrl)

        self._create_seg_plot("ecg", t2, height=240)
        self._create_seg_plot("pcg", t2, height=240)
        self.tabs.addTab(self.tab_seg, "Segmentation")

        # CWT tab (also hosts TF method choice between CWT / STFT)
        self.tab_cwt = QtWidgets.QWidget()
        t3 = QtWidgets.QVBoxLayout(self.tab_cwt)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("fmin"))
        self.cwt_fmin = QtWidgets.QLineEdit(str(DEFAULT_CWT_FMIN)); self.cwt_fmin.setMaximumWidth(90)
        self.cwt_fmin.setToolTip("Minimum frequency for CWT (Hz).")
        ctrl.addWidget(self.cwt_fmin)
        ctrl.addWidget(QtWidgets.QLabel("fmax"))
        self.cwt_fmax = QtWidgets.QLineEdit(str(DEFAULT_CWT_FMAX)); self.cwt_fmax.setMaximumWidth(90)
        self.cwt_fmax.setToolTip("Maximum frequency for CWT (Hz).")
        ctrl.addWidget(self.cwt_fmax)
        ctrl.addWidget(QtWidgets.QLabel("n_freqs"))
        self.cwt_nfreqs = QtWidgets.QLineEdit(str(DEFAULT_CWT_NFREQS)); self.cwt_nfreqs.setMaximumWidth(90)
        self.cwt_nfreqs.setToolTip("Number of frequency rows (vertical resolution).")
        ctrl.addWidget(self.cwt_nfreqs)

        ctrl.addWidget(QtWidgets.QLabel("Backend"))
        self.cwt_backend_combo = QtWidgets.QComboBox()
        self.cwt_backend_combo.setMaximumWidth(150)
        self.cwt_backend_combo.addItems(["pascal", "pywt", "scipy", "spectrogram"])
        self.cwt_backend_combo.setCurrentText("pascal")
        self.cwt_backend_combo.setToolTip("Choose CWT backend.")
        ctrl.addWidget(self.cwt_backend_combo)

        # Add TF method selection (CWT vs STFT)
        ctrl.addWidget(QtWidgets.QLabel("TF method"))
        self.tf_method_combo = QtWidgets.QComboBox()
        self.tf_method_combo.setMaximumWidth(140)
        self.tf_method_combo.addItems(["CWT", "STFT"])
        self.tf_method_combo.setCurrentText("CWT")
        self.tf_method_combo.setToolTip("Choose time-frequency method used before Threshold & CoG.")
        ctrl.addWidget(self.tf_method_combo)

        self.cwt_use_freqs_target = QtWidgets.QCheckBox("Use explicit freqs (linspace fmin->fmax)")
        self.cwt_use_freqs_target.setChecked(True)
        self.cwt_use_freqs_target.setToolTip("When checked, Pascal backend will use a freqs_target array = linspace(fmin,fmax,n_freqs).")
        ctrl.addWidget(self.cwt_use_freqs_target)

        ctrl.addWidget(QtWidgets.QLabel("col_count"))
        self.cwt_colcount_spin = QtWidgets.QSpinBox()
        self.cwt_colcount_spin.setRange(16, 2000)
        self.cwt_colcount_spin.setValue(DEFAULT_CWT_COLCOUNT)
        self.cwt_colcount_spin.setMaximumWidth(100)
        self.cwt_colcount_spin.setToolTip("Number of translation columns (time resolution).")
        ctrl.addWidget(self.cwt_colcount_spin)

        ctrl.addWidget(QtWidgets.QLabel("a0"))
        self.cwt_a0_spin = QtWidgets.QDoubleSpinBox()
        self.cwt_a0_spin.setDecimals(7)
        self.cwt_a0_spin.setRange(0.000001, 0.01)
        self.cwt_a0_spin.setSingleStep(0.000001)
        self.cwt_a0_spin.setValue(DEFAULT_PASCAL_A0)
        self.cwt_a0_spin.setMaximumWidth(120)
        self.cwt_a0_spin.setToolTip("Initial scale (a0) of Pascal CWT.")
        ctrl.addWidget(self.cwt_a0_spin)

        ctrl.addWidget(QtWidgets.QLabel("a_step"))
        self.cwt_astep_spin = QtWidgets.QDoubleSpinBox()
        self.cwt_astep_spin.setDecimals(8)
        self.cwt_astep_spin.setRange(0.0000001, 0.001)
        self.cwt_astep_spin.setSingleStep(0.0000001)
        self.cwt_astep_spin.setValue(DEFAULT_PASCAL_A_STEP)
        self.cwt_astep_spin.setMaximumWidth(120)
        self.cwt_astep_spin.setToolTip("Increment of scale per row (a_step).")
        ctrl.addWidget(self.cwt_astep_spin)

        ctrl.addStretch()
        self.method_label = QtWidgets.QLabel("CWT method: (not computed)")
        ctrl.addWidget(self.method_label)
        t3.addLayout(ctrl)

        self._create_cwt_canvas(t3)
        self.tabs.addTab(self.tab_cwt, "CWT")

        # STFT tab (new)
        self.tab_stft = QtWidgets.QWidget()
        t_stft = QtWidgets.QVBoxLayout(self.tab_stft)
        stft_ctrl = QtWidgets.QHBoxLayout()
        stft_ctrl.addWidget(QtWidgets.QLabel("nperseg"))
        self.stft_nperseg = QtWidgets.QSpinBox(); self.stft_nperseg.setRange(16, 8192); self.stft_nperseg.setValue(256); self.stft_nperseg.setMaximumWidth(100)
        stft_ctrl.addWidget(self.stft_nperseg)
        stft_ctrl.addWidget(QtWidgets.QLabel("noverlap"))
        self.stft_noverlap = QtWidgets.QSpinBox(); self.stft_noverlap.setRange(0, 8191); self.stft_noverlap.setValue(128); self.stft_noverlap.setMaximumWidth(100)
        stft_ctrl.addWidget(self.stft_noverlap)
        stft_ctrl.addWidget(QtWidgets.QLabel("nfft"))
        self.stft_nfft = QtWidgets.QSpinBox(); self.stft_nfft.setRange(32, 16384); self.stft_nfft.setValue(512); self.stft_nfft.setMaximumWidth(100)
        stft_ctrl.addWidget(self.stft_nfft)
        stft_ctrl.addStretch()
        t_stft.addLayout(stft_ctrl)
        # create canvases for STFT
        self._create_stft_canvas(t_stft)
        self.tabs.addTab(self.tab_stft, "STFT")

        # Threshold & CoG
        self.tab_thr = QtWidgets.QWidget()
        t4 = QtWidgets.QVBoxLayout(self.tab_thr)
        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel("S1 thr"))
        self.thr_s1 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s1.setMinimum(1); self.thr_s1.setMaximum(99); self.thr_s1.setValue(60); self.thr_s1.setEnabled(False)
        thr_row.addWidget(self.thr_s1); self.thr_s1_label = QtWidgets.QLabel("0.60"); thr_row.addWidget(self.thr_s1_label)
        thr_row.addWidget(QtWidgets.QLabel("S2 thr"))
        self.thr_s2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s2.setMinimum(1); self.thr_s2.setMaximum(99); self.thr_s2.setValue(10); self.thr_s2.setEnabled(False)
        thr_row.addWidget(self.thr_s2); self.thr_s2_label = QtWidgets.QLabel("0.10"); thr_row.addWidget(self.thr_s2_label)

        thr_row.addStretch()
        thr_row.addWidget(QtWidgets.QLabel("min_area"))
        self.thr_min_area_spin = QtWidgets.QSpinBox()
        self.thr_min_area_spin.setRange(0, 1000)
        self.thr_min_area_spin.setValue(0)
        self.thr_min_area_spin.setMaximumWidth(110)
        thr_row.addWidget(self.thr_min_area_spin)

        thr_row.addWidget(QtWidgets.QLabel("keep_top"))
        self.thr_keep_top_spin = QtWidgets.QSpinBox()
        self.thr_keep_top_spin.setRange(1, 20)
        self.thr_keep_top_spin.setValue(3)
        self.thr_keep_top_spin.setMaximumWidth(80)
        thr_row.addWidget(self.thr_keep_top_spin)

        t4.addLayout(thr_row)
        self._create_thr_canvas(t4)
        self.tabs.addTab(self.tab_thr, "Threshold & CoG")

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    # ---------------- helper functions to build canvases ----------------
    def _mk_figure_canvas(self, figsize=(8, 3.0)):
        """Create Matplotlib figure+canvas with dark figure background."""
        fig = Figure(figsize=figsize, facecolor=FIG_BG)
        canvas = FigureCanvas(fig)
        fig.patch.set_facecolor(FIG_BG)
        return fig, canvas

    def _style_axis_dark(self, ax):
        """Dark styling + persistent dashed gridlines on zoom/pan."""
        try:
            ax.set_facecolor(AX_BG)
            ax.tick_params(colors=FIG_FG, which='both')
            for spine in ax.spines.values():
                spine.set_color(FIG_FG)
            ax.xaxis.label.set_color(FIG_FG)
            ax.yaxis.label.set_color(FIG_FG)
            ax.title.set_color(FIG_FG)
            ax.grid(color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.6)
            try:
                def _on_xlim_changed(axobj):
                    axobj.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.6)
                def _on_ylim_changed(axobj):
                    axobj.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.6)
                ax.callbacks.connect('xlim_changed', _on_xlim_changed)
                ax.callbacks.connect('ylim_changed', _on_ylim_changed)
            except Exception:
                pass
        except Exception:
            pass

    def _wrapped_scroll_widget(self, canvas, toolbar=None, height=240):
        """
        Put toolbar + canvas inside a widget and wrap it into a QScrollArea.
        Use setWidgetResizable(True) and Expanding policies so scrollbars behave well
        across resize/minimize/maximize.
        """
        container = QtWidgets.QWidget()
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)
        if toolbar is not None:
            v.addWidget(toolbar)
        v.addWidget(canvas)

        # set canvas size policy to Expanding so it participates in layout resizing
        try:
            fig = getattr(canvas, 'figure', None)
            if fig is not None:
                dpi = fig.get_dpi()
                h_px = int(max(160, fig.get_figheight() * dpi))
                w_px = int(max(400, fig.get_figwidth() * dpi))
                # keep minimum size for readability but allow expansion/shrink on window resize
                canvas.setMinimumHeight(h_px)
                canvas.setMinimumWidth(w_px)
                canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
                canvas.updateGeometry()
        except Exception:
            pass

        scroll = QtWidgets.QScrollArea()
        # Important: make scroll area resizable to allow normal behavior on resize
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(height)
        # do NOT clamp maximumHeight here; allow tab layout to use space correctly
        return scroll

    def _create_pt_plot(self, which, parent_layout, height=260):
        if which == 'raw':
            self.fig_pt_raw, self.canvas_pt_raw = self._mk_figure_canvas(figsize=(10, 2.8))
            self.toolbar_pt_raw = NavigationToolbar(self.canvas_pt_raw, self)
            self.ax_pt_ecg_raw = self.fig_pt_raw.add_subplot(111)
            self._style_axis_dark(self.ax_pt_ecg_raw)
            scroll = self._wrapped_scroll_widget(self.canvas_pt_raw, self.toolbar_pt_raw, height=height)
            parent_layout.addWidget(scroll)
        elif which == 'proc':
            self.fig_pt_proc, self.canvas_pt_proc = self._mk_figure_canvas(figsize=(10, 2.4))
            self.toolbar_pt_proc = NavigationToolbar(self.canvas_pt_proc, self)
            self.ax_pt_ecg_proc = self.fig_pt_proc.add_subplot(111)
            self._style_axis_dark(self.ax_pt_ecg_proc)
            scroll = self._wrapped_scroll_widget(self.canvas_pt_proc, self.toolbar_pt_proc, height=height)
            parent_layout.addWidget(scroll)
        elif which == 'pcg':
            self.fig_pt_pcg, self.canvas_pt_pcg = self._mk_figure_canvas(figsize=(10, 2.8))
            self.toolbar_pt_pcg = NavigationToolbar(self.canvas_pt_pcg, self)
            self.ax_pt_pcg = self.fig_pt_pcg.add_subplot(111)
            self._style_axis_dark(self.ax_pt_pcg)
            scroll = self._wrapped_scroll_widget(self.canvas_pt_pcg, self.toolbar_pt_pcg, height=height)
            parent_layout.addWidget(scroll)
        else:
            raise ValueError("unknown PT plot")

    def _create_seg_plot(self, which, parent_layout, height=240):
        if which == 'ecg':
            self.fig_seg_ecg, self.canvas_seg_ecg = self._mk_figure_canvas(figsize=(10, 2.6))
            self.toolbar_seg_ecg = NavigationToolbar(self.canvas_seg_ecg, self)
            self.ax_seg_ecg = self.fig_seg_ecg.add_subplot(111)
            self._style_axis_dark(self.ax_seg_ecg)
            scroll = self._wrapped_scroll_widget(self.canvas_seg_ecg, self.toolbar_seg_ecg, height=height)
            parent_layout.addWidget(scroll)
        elif which == 'pcg':
            self.fig_seg_pcg, self.canvas_seg_pcg = self._mk_figure_canvas(figsize=(10, 2.6))
            self.toolbar_seg_pcg = NavigationToolbar(self.canvas_seg_pcg, self)
            self.ax_seg_pcg = self.fig_seg_pcg.add_subplot(111)
            self._style_axis_dark(self.ax_seg_pcg)
            scroll = self._wrapped_scroll_widget(self.canvas_seg_pcg, self.toolbar_seg_pcg, height=height)
            parent_layout.addWidget(scroll)
        else:
            raise ValueError("unknown seg plot")

    def _create_cwt_canvas(self, parent_layout):
        self.fig_cwt, self.canvas_cwt = self._mk_figure_canvas(figsize=(10, 4.5))
        # --- add separate 3D CWT canvas (so 2D and 3D are both visible + scrollable) ---
        self.fig_cwt_3d, self.canvas_cwt_3d = self._mk_figure_canvas(figsize=(10, 4.0))
        self.toolbar_cwt_3d = NavigationToolbar(self.canvas_cwt_3d, self)
        try:
            # try to create a 3D axis
            self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111, projection='3d')
            # apply dark-ish pane colors (best-effort)
            try:
                self.ax_cwt_3d.w_xaxis.set_pane_color((0.03,0.07,0.14,1.0))
                self.ax_cwt_3d.w_yaxis.set_pane_color((0.03,0.07,0.14,1.0))
                self.ax_cwt_3d.w_zaxis.set_pane_color((0.03,0.07,0.14,1.0))
            except Exception:
                pass
        except Exception:
            # fallback: some environments might not support 3D subplots; create a normal 2D axis instead
            self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111)
        # wrap into a scroll widget (same helper used elsewhere)
        scroll3d = self._wrapped_scroll_widget(self.canvas_cwt_3d, self.toolbar_cwt_3d, height=420)
        parent_layout.addWidget(scroll3d)

        self.toolbar_cwt = NavigationToolbar(self.canvas_cwt, self)
        self.ax_cwt = self.fig_cwt.add_subplot(111)
        self._style_axis_dark(self.ax_cwt)
        scroll = self._wrapped_scroll_widget(self.canvas_cwt, self.toolbar_cwt, height=420)
        parent_layout.addWidget(scroll)

    def _create_thr_canvas(self, parent_layout):
        self.fig_thr, self.canvas_thr = self._mk_figure_canvas(figsize=(10, 4.5))
        self.toolbar_thr = NavigationToolbar(self.canvas_thr, self)
        self.ax_thr = self.fig_thr.add_subplot(111)
        self._style_axis_dark(self.ax_thr)
        scroll = self._wrapped_scroll_widget(self.canvas_thr, self.toolbar_thr, height=420)
        parent_layout.addWidget(scroll)

    def _create_stft_canvas(self, parent_layout):
        # 2D STFT canvas
        self.fig_stft_2d, self.canvas_stft_2d = self._mk_figure_canvas(figsize=(10, 3.8))
        self.toolbar_stft_2d = NavigationToolbar(self.canvas_stft_2d, self)
        self.ax_stft_2d = self.fig_stft_2d.add_subplot(111)
        self._style_axis_dark(self.ax_stft_2d)
        scroll2d = self._wrapped_scroll_widget(self.canvas_stft_2d, self.toolbar_stft_2d, height=360)
        parent_layout.addWidget(scroll2d)

        # 3D STFT canvas
        self.fig_stft_3d, self.canvas_stft_3d = self._mk_figure_canvas(figsize=(10, 4.0))
        self.toolbar_stft_3d = NavigationToolbar(self.canvas_stft_3d, self)
        try:
            self.ax_stft_3d = self.fig_stft_3d.add_subplot(111, projection='3d')
            # try to set pane color similar to dark theme
            try:
                self.ax_stft_3d.w_xaxis.set_pane_color((0.03,0.07,0.14,1.0))
                self.ax_stft_3d.w_yaxis.set_pane_color((0.03,0.07,0.14,1.0))
                self.ax_stft_3d.w_zaxis.set_pane_color((0.03,0.07,0.14,1.0))
            except Exception:
                pass
        except Exception:
            # fallback to 2D if 3D not supported
            self.ax_stft_3d = self.fig_stft_3d.add_subplot(111)
        scroll3d = self._wrapped_scroll_widget(self.canvas_stft_3d, self.toolbar_stft_3d, height=420)
        parent_layout.addWidget(scroll3d)

    # ---------------- connect signals ----------------
    def _connect_signals(self):
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.clear_btn.clicked.connect(self.clear_all)
        self.quit_btn.clicked.connect(QtWidgets.QApplication.quit)
        self.swap_btn.clicked.connect(self._on_swap_channels)
        self.pt_detect_btn.clicked.connect(self._on_run_pt)
        self.show_pt_pipeline_btn.clicked.connect(self._on_show_pt_pipeline)
        self.seg_beat_slider.valueChanged.connect(self._on_seg_beat_changed)
        self.seg_beat_slider.sliderReleased.connect(self._on_seg_slider_released)

        self.cwt_fmin.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_fmax.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_nfreqs.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_backend_combo.currentIndexChanged.connect(self._on_cwt_backend_changed)
        self.cwt_use_freqs_target.stateChanged.connect(self._on_cwt_params_changed)
        self.cwt_colcount_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_a0_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_astep_spin.valueChanged.connect(self._on_cwt_params_changed)

        # TF method selection reuses same handler to trigger recompute
        self.tf_method_combo.currentIndexChanged.connect(self._on_cwt_backend_changed)

        # STFT params -> recompute
        self.stft_nperseg.valueChanged.connect(self._on_cwt_params_changed)
        self.stft_noverlap.valueChanged.connect(self._on_cwt_params_changed)
        self.stft_nfft.valueChanged.connect(self._on_cwt_params_changed)

        self.thr_s1.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_s2.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_min_area_spin.valueChanged.connect(self._on_thr_params_changed)
        self.thr_keep_top_spin.valueChanged.connect(self._on_thr_params_changed)

        self.save_btn.clicked.connect(self._on_save_results)

    # ---------------- helpers unchanged ----------------
    def _ensure_scalogram_orientation_and_axes(self, scal, freqs, times):
        scal2 = np.asarray(scal)
        if scal2.ndim != 2:
            raise ValueError("scalogram must be 2D")
        nrows, ncols = scal2.shape

        freqs_in = np.asarray(freqs) if freqs is not None else None
        times_in = np.asarray(times) if times is not None else None

        try:
            if freqs_in is not None and freqs_in.size == ncols and freqs_in.size != nrows:
                scal2 = scal2.T
                nrows, ncols = scal2.shape
                freqs_in = np.asarray(freqs_in)
        except Exception:
            pass

        if freqs_in is None or freqs_in.size == 0:
            freqs2 = np.linspace(0.0, 1.0, nrows)
        else:
            if freqs_in.size != nrows:
                freqs2 = np.linspace(float(freqs_in[0]), float(freqs_in[-1]), nrows)
            else:
                freqs2 = freqs_in.copy()

        if times_in is None or times_in.size == 0:
            times2 = np.linspace(0.0, float(ncols - 1) / float(self.fs), ncols)
        else:
            if times_in.size != ncols:
                t0 = float(times_in[0]) if times_in.size > 0 else 0.0
                tend = float(times_in[-1]) if times_in.size > 1 else (t0 + float(ncols - 1)/float(self.fs))
                times2 = np.linspace(t0, tend, ncols)
            else:
                times2 = times_in.copy()

        if freqs2[0] > freqs2[-1]:
            scal2 = scal2[::-1, :]
            freqs2 = freqs2[::-1]

        return scal2, freqs2, times2

    # ---------------- core actions: load, PT, segmentation (logic preserved) ----------------
    def clear_all(self):
        self.record_p_signal = None
        self.fs = DEFAULT_FS
        self.sig_names = []
        self.ecg_idx = None
        self.pcg_idx = None
        self.r_peaks = np.array([], dtype=int)
        self.current_segment = None
        self.segment_bounds = (0, 0)
        self.current_scalogram = None
        self.current_freqs = None
        self.current_times = None
        self.current_cwt_method = None
        self.current_masks = {}
        self.current_cogs = {}

        self.save_btn.setEnabled(False)
        self.swap_btn.setEnabled(False)
        self.pt_detect_btn.setEnabled(False)
        self.show_pt_pipeline_btn.setEnabled(False)
        self.seg_beat_slider.setEnabled(False)
        self.thr_s1.setEnabled(False)
        self.thr_s2.setEnabled(False)

        try:
            if self.cwt_cb is not None:
                self.cwt_cb.remove()
                self.cwt_cb = None
        except Exception:
            self.cwt_cb = None
        try:
            if self.cwt_cax is not None:
                try: self.fig_cwt.delaxes(self.cwt_cax)
                except Exception: pass
                self.cwt_cax = None
        except Exception:
            pass

        try:
            if self.thr_cb is not None:
                self.thr_cb.remove()
                self.thr_cb = None
        except Exception:
            self.thr_cb = None
        try:
            if self.thr_cax is not None:
                try: self.fig_thr.delaxes(self.thr_cax)
                except Exception: pass
                self.thr_cax = None
        except Exception:
            pass

        self.last_thr_contours = []
        try:
            for art in getattr(self, 'last_thr_scat', []):
                try: art.remove()
                except Exception: pass
        except Exception:
            pass
        self.last_thr_scat = []
        self.cwt_im = None
        self.thr_im = None

        for fig in (getattr(self, 'fig_pt_raw', None),
                    getattr(self, 'fig_pt_proc', None),
                    getattr(self, 'fig_pt_pcg', None),
                    getattr(self, 'fig_seg_ecg', None),
                    getattr(self, 'fig_seg_pcg', None),
                    getattr(self, 'fig_cwt', None),
                    getattr(self, 'fig_thr', None),
                    getattr(self, 'fig_stft_2d', None),
                    getattr(self, 'fig_stft_3d', None)):
            try:
                if fig is not None:
                    fig.clf()
            except Exception:
                pass

        try:
            if getattr(self, 'fig_pt_raw', None) is not None:
                self.ax_pt_ecg_raw = self.fig_pt_raw.add_subplot(111); self._style_axis_dark(self.ax_pt_ecg_raw)
        except Exception: pass
        try:
            if getattr(self, 'fig_pt_proc', None) is not None:
                self.ax_pt_ecg_proc = self.fig_pt_proc.add_subplot(111); self._style_axis_dark(self.ax_pt_ecg_proc)
        except Exception: pass
        try:
            if getattr(self, 'fig_pt_pcg', None) is not None:
                self.ax_pt_pcg = self.fig_pt_pcg.add_subplot(111); self._style_axis_dark(self.ax_pt_pcg)
        except Exception: pass
        try:
            if getattr(self, 'fig_seg_ecg', None) is not None:
                self.ax_seg_ecg = self.fig_seg_ecg.add_subplot(111); self._style_axis_dark(self.ax_seg_ecg)
        except Exception: pass
        try:
            if getattr(self, 'fig_seg_pcg', None) is not None:
                self.ax_seg_pcg = self.fig_seg_pcg.add_subplot(111); self._style_axis_dark(self.ax_seg_pcg)
        except Exception: pass
        try:
            if getattr(self, 'fig_cwt', None) is not None:
                self.ax_cwt = self.fig_cwt.add_subplot(111); self._style_axis_dark(self.ax_cwt)
        except Exception: pass
        try:
            if getattr(self, 'fig_thr', None) is not None:
                self.ax_thr = self.fig_thr.add_subplot(111); self._style_axis_dark(self.ax_thr)
        except Exception: pass

        for c in (getattr(self, 'canvas_pt_raw', None),
                  getattr(self, 'canvas_pt_proc', None),
                  getattr(self, 'canvas_pt_pcg', None),
                  getattr(self, 'canvas_seg_ecg', None),
                  getattr(self, 'canvas_seg_pcg', None),
                  getattr(self, 'canvas_cwt', None),
                  getattr(self, 'canvas_thr', None),
                  getattr(self, 'canvas_stft_2d', None),
                  getattr(self, 'canvas_stft_3d', None)):
            try:
                if c is not None:
                    c.draw_idle()
            except Exception:
                pass

        self.record_label.setText("No record loaded")
        self.method_label.setText("CWT method: (not computed)")
        self.status.showMessage("Cleared")

    def _on_load_clicked(self):
        self.clear_all()
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .hea or .dat", os.getcwd(), "Header/Dat Files (*.hea *.dat)")
        if not fname:
            return
        base = os.path.splitext(fname)[0]
        try:
            info = try_load_record(base)
        except Exception as e:
            self.status.showMessage(f"Load failed: {e}")
            return
        self.record_p_signal = info['p_signal']
        self.fs = int(info.get('fs', DEFAULT_FS))
        self.sig_names = info.get('sig_name', []) or []
        chmap = info.get('channel_map', {})
        if chmap:
            self.ecg_idx = int(chmap.get('ecg', 1 if self.record_p_signal.shape[1] > 1 else 0))
            self.pcg_idx = int(chmap.get('pcg', 0))
        else:
            nch = self.record_p_signal.shape[1]
            if nch >= 2:
                self.pcg_idx = 0; self.ecg_idx = 1
            else:
                self.pcg_idx = 0; self.ecg_idx = None
        if self.ecg_idx == self.pcg_idx and self.record_p_signal.shape[1] >= 2:
            self.pcg_idx, self.ecg_idx = 0, 1
        self.record_label.setText(f"Loaded: {os.path.basename(base)}  fs={self.fs} ch={self.record_p_signal.shape[1]}")
        self.status.showMessage("Record loaded. Run Pan-Tompkins.")
        self.pt_detect_btn.setEnabled(True)
        self.swap_btn.setEnabled(True)

    def _on_swap_channels(self):
        if self.record_p_signal is None:
            return
        self.ecg_idx, self.pcg_idx = self.pcg_idx, self.ecg_idx
        self.status.showMessage(f"Swapped: ECG={self.ecg_idx}, PCG={self.pcg_idx}")
        if self.r_peaks is not None and len(self.r_peaks) > 1:
            self._on_run_pt()
        
    def _on_run_pt(self):
        """Run Pan-Tompkins detection and update PT plots + segmentation slider."""
        if self.record_p_signal is None or self.ecg_idx is None:
            self.status.showMessage("Load a record and ensure ECG channel assigned.")
            return
        ecg = self.record_p_signal[:, self.ecg_idx].astype(float)
        try:
            r_peaks = detect_r_peaks_with_fallback(ecg, fs=self.fs, debug=False)
        except Exception as e:
            self.status.showMessage(f"PT detection error: {e}")
            print(traceback.format_exc())
            return

        if r_peaks is None or len(r_peaks) < 2:
            self.status.showMessage("Too few R peaks detected. Try swapping channels or inspect data.")
            return

        self.r_peaks = np.array(r_peaks, dtype=int)
        self.show_pt_pipeline_btn.setEnabled(True)

        # processed signals for plotting (best-effort)
        try:
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * self.fs
            b, a = butter(3, [5.0/nyq, 15.0/nyq], btype='band')
            ecg_filt = filtfilt(b, a, ecg)
            deriv = np.diff(ecg_filt, prepend=ecg_filt[0])
            squared = deriv ** 2
            win = max(1, int(round(150.0 / 1000.0 * self.fs)))
            mwi = np.convolve(squared, np.ones(win)/win, mode='same')
        except Exception:
            ecg_filt = ecg.copy()
            mwi = np.abs(ecg)

        pcg = self.record_p_signal[:, self.pcg_idx].astype(float)

        # Plot PT tab
        self.ax_pt_ecg_raw.clear()
        t = np.arange(ecg.size) / float(self.fs)
        self.ax_pt_ecg_raw.plot(t, ecg, linewidth=0.6)
        self.ax_pt_ecg_raw.scatter(self.r_peaks / float(self.fs), ecg[self.r_peaks], c='r', s=10, label='R-peaks')
        self.ax_pt_ecg_raw.set_ylabel("ECG (raw)")
        self.ax_pt_ecg_raw.legend(fontsize='small')
        self.ax_pt_ecg_raw.set_facecolor(AX_BG)

        self.ax_pt_ecg_proc.clear()
        self.ax_pt_ecg_proc.plot(t, ecg_filt, linewidth=0.6, label='filtered')
        if mwi.max() != 0:
            self.ax_pt_ecg_proc.plot(t, mwi / (mwi.max() + 1e-12), label='MWI (norm)', alpha=0.8)
        self.ax_pt_ecg_proc.set_ylabel("Processed")
        self.ax_pt_ecg_proc.legend(fontsize='small')
        self.ax_pt_ecg_proc.set_facecolor(AX_BG)

        self.ax_pt_pcg.clear()
        t_pcg = np.arange(pcg.size) / float(self.fs)
        self.ax_pt_pcg.plot(t_pcg, pcg, linewidth=0.6)
        self.ax_pt_pcg.set_ylabel("PCG (raw)")
        self.ax_pt_pcg.set_xlabel("Time [s]")
        self.ax_pt_pcg.set_facecolor(AX_BG)

        try:
            self.fig_pt.tight_layout()
        except Exception:
            pass
        self.canvas_pt_raw.draw_idle()
        self.canvas_pt_proc.draw_idle()
        self.canvas_pt_pcg.draw_idle()

        # segmentation slider
        n_beats = max(1, len(self.r_peaks) - 1)
        max_idx = max(0, n_beats - 1)
        self.seg_beat_slider.setMaximum(max_idx)
        self.seg_beat_slider.setEnabled(True)

        # --- NEW: choose a "clean" beat as default using choose_clean_beat(...) ---
        try:
            chosen_idx = choose_clean_beat(pcg, self.r_peaks, fs=self.fs)
            chosen_idx = int(chosen_idx)
        except Exception:
            chosen_idx = 0
        chosen_idx = max(0, min(chosen_idx, max_idx))
        self.seg_beat_slider.setValue(chosen_idx)
        self.seg_beat_label.setText(str(chosen_idx))

        # compute initial segment downstream
        self._update_segment_and_downstream(chosen_idx)
        self.status.showMessage(f"Pan-Tompkins: {len(self.r_peaks)} beats detected. (selected beat {chosen_idx})")

    def _embed_figures_in_dialog(self, fig_nums, title="Pan-Tompkins pipeline"):
        if not fig_nums:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1100, 800)
        dlg.setStyleSheet(QT_STYLE)

        main_v = QtWidgets.QVBoxLayout(dlg)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setSpacing(12)

        for num in fig_nums:
            try:
                fig = plt.figure(num)
            except Exception:
                continue
            # ensure dark figure patch and dark axes styling
            try:
                fig.patch.set_facecolor(FIG_BG)
            except Exception:
                pass
            for ax in getattr(fig, 'axes', []):
                try:
                    ax.set_facecolor(AX_BG)
                    ax.tick_params(colors=FIG_FG, which='both')
                    ax.xaxis.label.set_color(FIG_FG); ax.yaxis.label.set_color(FIG_FG)
                    ax.title.set_color(FIG_FG)
                    ax.grid(True, linestyle=':', color=GRID_COLOR, alpha=0.6)
                    for spine in ax.spines.values():
                        spine.set_color(FIG_FG)
                    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                        try:
                            lbl.set_color(FIG_FG)
                        except Exception:
                            pass
                    try:
                        def _on_xlim(axobj): axobj.grid(True, linestyle=':', color=GRID_COLOR, alpha=0.6)
                        def _on_ylim(axobj): axobj.grid(True, linestyle=':', color=GRID_COLOR, alpha=0.6)
                        ax.callbacks.connect('xlim_changed', _on_xlim)
                        ax.callbacks.connect('ylim_changed', _on_ylim)
                    except Exception:
                        pass
                except Exception:
                    pass

            try:
                fig.tight_layout()
            except Exception:
                pass

            canvas = FigureCanvas(fig)
            # sizing: set minimum size but allow expansion so scroll works correctly
            try:
                dpi = fig.get_dpi()
                h_px = int(max(200, fig.get_figheight() * dpi))
                w_px = int(max(480, fig.get_figwidth() * dpi))
                canvas.setMinimumHeight(h_px)
                canvas.setMinimumWidth(w_px)
                canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
                canvas.updateGeometry()
            except Exception:
                pass

            toolbar = NavigationToolbar(canvas, self)
            w = QtWidgets.QWidget()
            wl = QtWidgets.QVBoxLayout(w)
            wl.setContentsMargins(2,2,2,2)
            wl.addWidget(toolbar)
            wl.addWidget(canvas)
            vbox.addWidget(w)
            try:
                canvas.draw()
            except Exception:
                pass

        scroll.setWidget(container)
        main_v.addWidget(scroll)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        main_v.addLayout(btn_row)
        dlg.show()

    def _on_show_pt_pipeline(self):
        if self.record_p_signal is None or self.ecg_idx is None:
            self.status.showMessage("Load record first.")
            return
        ecg = self.record_p_signal[:, self.ecg_idx].astype(float)

        existing = set(plt.get_fignums())
        try:
            plot_pt_pipeline(ecg, fs=self.fs, r_peaks=self.r_peaks)
        except Exception as e:
            self.status.showMessage(f"plot_pt_pipeline error: {e}")
            print(traceback.format_exc())
            return

        new_set = set(plt.get_fignums()) - existing
        new_list = sorted(list(new_set))
        if not new_list:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("PT pipeline")
            msg.setText("plot_pt_pipeline did not create new matplotlib figures to embed.")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg.exec()
            return
        self._embed_figures_in_dialog(new_list, title="Pan-Tompkins pipeline (dark)")

    # segmentation & downstream (logic preserved) ----------------
    def _on_seg_beat_changed(self, val):
        self.seg_beat_label.setText(str(val))

    def _on_seg_slider_released(self):
        idx = int(self.seg_beat_slider.value())
        self._update_segment_and_downstream(idx)

    def _update_segment_and_downstream(self, beat_idx: int):
        if self.record_p_signal is None or self.r_peaks is None or len(self.r_peaks) < 2:
            return
        pcg = self.record_p_signal[:, self.pcg_idx].astype(float)
        try:
            seg, start, end = segment_one_cycle(pcg, self.r_peaks, idx=beat_idx, pad_ms=50.0, fs=self.fs)
        except Exception:
            r0 = int(self.r_peaks[beat_idx]); r1 = int(self.r_peaks[beat_idx+1])
            pad = int(round(0.05 * self.fs))
            start = max(0, r0 - pad); end = min(len(pcg), r1 + pad)
            seg = pcg[start:end]

        self.current_segment = seg
        self.segment_bounds = (start, end)

        try:
            self.ax_seg_ecg.clear()
            if self.ecg_idx is not None:
                ecg = self.record_p_signal[:, self.ecg_idx].astype(float)
                r0 = int(self.r_peaks[beat_idx])
                window = int(round(0.3 * self.fs))
                lo = max(0, r0 - window); hi = min(len(ecg) - 1, r0 + window)
                t_ecg = np.arange(lo, hi + 1) / float(self.fs)
                self.ax_seg_ecg.plot(t_ecg, ecg[lo:hi + 1])
                self.ax_seg_ecg.axvline(r0 / float(self.fs), color='r', linestyle='--', linewidth=0.8)
                self.ax_seg_ecg.set_title("ECG around selected R")
                self.ax_seg_ecg.set_ylabel("ECG")
            else:
                self.ax_seg_ecg.text(0.5, 0.5, "No ECG channel", transform=self.ax_seg_ecg.transAxes)
            self._style_axis_dark(self.ax_seg_ecg)
            try: self.fig_seg_ecg.tight_layout()
            except Exception: pass
            self.canvas_seg_ecg.draw_idle()
        except Exception:
            pass

        try:
            self.ax_seg_pcg.clear()
            t_seg = np.arange(start, end) / float(self.fs)
            self.ax_seg_pcg.plot(t_seg, seg)
            self.ax_seg_pcg.set_title("PCG segment")
            self.ax_seg_pcg.set_ylabel("PCG")
            self.ax_seg_pcg.set_xlabel("Time [s]")
            self._style_axis_dark(self.ax_seg_pcg)
            try: self.fig_seg_pcg.tight_layout()
            except Exception: pass
            self.canvas_seg_pcg.draw_idle()
        except Exception:
            pass

        # read CWT params (safe fallback)
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            fmin, fmax, nfreqs = DEFAULT_CWT_FMIN, DEFAULT_CWT_FMAX, DEFAULT_CWT_NFREQS

        # Determine TF method choice (CWT or STFT)
        tf_choice = 'cwt'
        try:
            tf_choice = str(self.tf_method_combo.currentText()).strip().lower()
        except Exception:
            tf_choice = 'cwt'

        if tf_choice == 'stft':
            # gather STFT params
            nperseg = int(self.stft_nperseg.value()) if hasattr(self, 'stft_nperseg') else None
            noverlap = int(self.stft_noverlap.value()) if hasattr(self, 'stft_noverlap') else None
            nfft = int(self.stft_nfft.value()) if hasattr(self, 'stft_nfft') else None
            if compute_stft is None:
                self.status.showMessage("STFT module not available (STFT_PCG.py missing).")
                return
            try:
                Sxx, freqs, times_rel, method = compute_stft(seg, fs=self.fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann', mode='magnitude')
            except Exception as e:
                self.status.showMessage(f"STFT error: {e}")
                print(traceback.format_exc())
                return
        else:
            # existing CWT path (unchanged)
            backend_sel = str(self.cwt_backend_combo.currentText()).strip().lower()
            compute_kwargs = {}
            if backend_sel == 'pascal':
                compute_kwargs['a0'] = float(self.cwt_a0_spin.value())
                compute_kwargs['a_step'] = float(self.cwt_astep_spin.value())
                compute_kwargs['col_count'] = int(self.cwt_colcount_spin.value())
                if self.cwt_use_freqs_target.isChecked():
                    compute_kwargs['freqs_target'] = np.linspace(fmin, fmax, nfreqs)
            try:
                Sxx, freqs, times_rel, method = compute_cwt(seg, fs=self.fs, fmin=fmin, fmax=fmax, n_freqs=nfreqs, backend=backend_sel, **compute_kwargs)
            except Exception as e:
                self.status.showMessage(f"CWT error: {e}")
                print(traceback.format_exc())
                return

        # adjust times (relative -> absolute) if necessary
        try:
            seg_dur = float(len(self.current_segment)) / float(self.fs) if (self.current_segment is not None) else None
            if times_rel is not None:
                times_rel = np.asarray(times_rel, dtype=float)
                if seg_dur is not None and times_rel.size > 0:
                    if times_rel.max() > (seg_dur * 1.2):
                        # times were probably in samples, convert to seconds
                        times_rel = times_rel / float(self.fs)
        except Exception:
            pass

        start_sample = int(self.segment_bounds[0])
        times_abs = times_rel + start_sample / float(self.fs) if (times_rel is not None and len(times_rel) > 0) else None

        try:
            scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(Sxx, freqs, times_abs)
        except Exception:
            self.status.showMessage("TF returned invalid scalogram shape")
            return

        # clear previous colorbars if needed (for cwt)
        self.cwt_im = None
        if getattr(self, 'cwt_cb', None) is not None:
            try: self.cwt_cb.remove()
            except Exception: pass
            self.cwt_cb = None
        if getattr(self, 'cwt_cax', None) is not None:
            try: self.fig_cwt.delaxes(self.cwt_cax)
            except Exception: pass
            self.cwt_cax = None

        # set current scalogram so thresholding works unchanged
        self.current_scalogram = scal2
        self.current_freqs = freqs2
        self.current_times = times2

        self.current_cwt_method = method
        self.method_label.setText(f"TF method: {method} (mode: {tf_choice})")

        self.current_masks = {}
        self.current_cogs = {}

        # update masks and CoG, plot TF tab(s), enable threshold controls
        self._update_masks_and_cogs()
        # plot the chosen TF tab(s)
        if tf_choice == 'stft':
            try:
                self._plot_stft_tab()
            except Exception:
                pass
        else:
            try:
                self._plot_cwt_tab()
            except Exception:
                pass

        self.thr_s1.setEnabled(True); self.thr_s2.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status.showMessage(f"Segment updated. TF method: {method}")
        self._plot_threshold_tab()

    # ---------- CWT plotting ----------
    def _plot_cwt_tab(self):
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            self.ax_cwt.clear()
            try:
                for ax in list(self.fig_cwt.axes):
                    if ax is not self.ax_cwt:
                        self.fig_cwt.delaxes(ax)
            except Exception:
                pass
            if getattr(self, 'cwt_cb', None) is not None:
                try: self.cwt_cb.remove()
                except Exception: pass
                self.cwt_cb = None
            self.cwt_im = None
            self.canvas_cwt.draw_idle()
            return

        scal = self.current_scalogram
        freqs = self.current_freqs
        times = self.current_times

        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(scal, freqs, times)
        except Exception:
            self.ax_cwt.clear()
            self.canvas_cwt.draw_idle()
            return

        self.current_scalogram = scal; self.current_freqs = freqs; self.current_times = times

        try:
            for ax in list(self.fig_cwt.axes):
                if ax is not self.ax_cwt:
                    self.fig_cwt.delaxes(ax)
        except Exception:
            pass
        if getattr(self, 'cwt_cb', None) is not None:
            try: self.cwt_cb.remove()
            except Exception: pass
            self.cwt_cb = None

        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        try:
            self.ax_cwt.clear()
            self.cwt_im = self.ax_cwt.imshow(scal, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            try:
                divider = make_axes_locatable(self.ax_cwt)
                cax = divider.append_axes("right", size="3%", pad=0.06)
                self.cwt_cb = self.fig_cwt.colorbar(self.cwt_im, cax=cax)
                self.cwt_cax = cax
            except Exception:
                self.cwt_cb = None
                self.cwt_cax = None
        except Exception:
            self.canvas_cwt.draw_idle()
            return

        try:
            self.ax_cwt.set_xlim(times[0], times[-1])
            self.ax_cwt.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass

        self.ax_cwt.set_xlabel("Time [s]"); self.ax_cwt.set_ylabel("Frequency [Hz]")
        self.ax_cwt.set_title(f"Scalogram ({self.current_cwt_method})")
        self._style_axis_dark(self.ax_cwt)
        try: self.fig_cwt.tight_layout()
        except Exception: pass
        self.canvas_cwt.draw_idle()
        
    # --- paste these methods inside your PCGAnalyzerGUI class ---

    def _style_3d_axis(self, ax):
        """Apply dark theme + readable ticks for 3D axis (best-effort)."""
        try:
            # face / pane colors (matplotlib 3D pane API)
            try:
                ax.w_xaxis.set_pane_color((0.03,0.07,0.14,1.0))
                ax.w_yaxis.set_pane_color((0.03,0.07,0.14,1.0))
                ax.w_zaxis.set_pane_color((0.03,0.07,0.14,1.0))
            except Exception:
                # older matplotlib fallback
                try:
                    ax.set_facecolor((0.03,0.07,0.14))
                except Exception:
                    pass
            # tick / label colours
            try:
                for lbl in (ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels() + ax.zaxis.get_ticklabels()):
                    lbl.set_color('#DDEFF8')
                ax.xaxis.label.set_color('#DDEFF8'); ax.yaxis.label.set_color('#DDEFF8'); ax.zaxis.label.set_color('#DDEFF8')
                ax.title.set_color('#EAF6FF')
            except Exception:
                pass
            # grid lines (dashed, slightly visible)
            try:
                ax.grid(True, linestyle=':', color=(0.08,0.12,0.2), linewidth=0.5)
            except Exception:
                pass
        except Exception:
            pass

    def _plot_cwt_3d(self, max_surface_points=120000, cmap='viridis', elev=35, azim=-60, use_db=True):
        """
        Plot a 3D surface of the current scalogram (time x freq x magnitude).
        - max_surface_points: approximate maximum number of cells (freq*time) to draw for surface (to keep UI responsive).
        - use_db: convert magnitude to dB for better dynamic range (default True).
        This method will create fig/canvas/ax attributes if they do not exist (non-destructive).
        """
        try:
            if self.current_scalogram is None:
                # nothing to plot
                return
            # ensure attributes for 3D canvas exist (safe: won't remove existing layout)
            if not hasattr(self, 'fig_cwt_3d') or getattr(self, 'fig_cwt_3d', None) is None:
                try:
                    # create a Matplotlib Figure + Canvas for 3D (best-effort)
                    self.fig_cwt_3d = Figure(figsize=(9, 4), facecolor=(0.03,0.07,0.14))
                    self.canvas_cwt_3d = FigureCanvas(self.fig_cwt_3d)
                    self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111, projection='3d')
                except Exception:
                    # fallback: try to reuse existing 2D fig if creation fails
                    try:
                        self.fig_cwt_3d = self.fig_cwt
                        self.canvas_cwt_3d = self.canvas_cwt
                        # add 3D axis if not present
                        self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111, projection='3d')
                    except Exception:
                        # give up gracefully
                        return

            scal = np.asarray(self.current_scalogram)
            freqs = np.asarray(self.current_freqs) if self.current_freqs is not None else None
            times = np.asarray(self.current_times) if self.current_times is not None else None

            # try to reuse your _ensure_scalogram_orientation_and_axes if present
            try:
                if hasattr(self, '_ensure_scalogram_orientation_and_axes'):
                    scal, freqs, times = self._ensure_scalogram_orientation_and_axes(scal, freqs, times)
            except Exception:
                # attempt a minimal correction
                if scal.ndim != 2:
                    raise ValueError("scalogram must be 2D for 3D plotting")
                nrows, ncols = scal.shape
                if freqs is None or len(freqs) != nrows:
                    freqs = np.linspace(0.0, 1.0*nrows, nrows)
                if times is None or len(times) != ncols:
                    times = np.linspace(0.0, (ncols-1)/float(self.fs if hasattr(self,'fs') else 1.0), ncols)

            # Build mesh
            T, F = np.meshgrid(times, freqs)
            Z = np.abs(scal)

            # optional convert to dB for improved visual dynamic range
            if use_db:
                Z_plot = 20.0 * np.log10(Z + 1e-12)
            else:
                Z_plot = Z

            # subsample if too many points
            n_total = Z_plot.size
            if n_total > max_surface_points:
                # downsample factor approx sqrt ratio
                r = float(n_total) / float(max_surface_points)
                factor = int(np.ceil(np.sqrt(r)))
                # pick subsampled indices
                f_idx = np.arange(0, Z_plot.shape[0], factor)
                t_idx = np.arange(0, Z_plot.shape[1], factor)
                Ts = T[np.ix_(f_idx, t_idx)]
                Fs = F[np.ix_(f_idx, t_idx)]
                Zs = Z_plot[np.ix_(f_idx, t_idx)]
            else:
                Ts, Fs, Zs = T, F, Z_plot

            # clear axis and plot
            try:
                self.ax_cwt_3d.cla()
            except Exception:
                pass

            try:
                surf = self.ax_cwt_3d.plot_surface(Ts, Fs, Zs,
                                                   rstride=1, cstride=1, cmap=cmap,
                                                   linewidth=0, antialiased=True)
                # attach colorbar in figure without shrinking axis too much
                try:
                    # Remove previous colorbar axes if exists
                    if hasattr(self, 'cwt_3d_cb_ax') and (self.cwt_3d_cb_ax in self.fig_cwt_3d.axes):
                        try:
                            self.fig_cwt_3d.delaxes(self.cwt_3d_cb_ax)
                        except Exception:
                            pass
                    # create a new small axes for colorbar
                    self.cwt_3d_cb_ax = self.fig_cwt_3d.add_axes([0.92, 0.15, 0.02, 0.7])
                    self.fig_cwt_3d.colorbar(surf, cax=self.cwt_3d_cb_ax)
                except Exception:
                    pass
            except Exception:
                # fallback: contour / pcolormesh drawing for 2D-like representation inside 3D axis
                try:
                    self.ax_cwt_3d.contourf(Ts, Fs, Zs, zdir='z', offset=Zs.min(), cmap=cmap)
                except Exception:
                    pass

            # labels, view, styling
            try:
                self.ax_cwt_3d.set_xlabel("Time [s]")
                self.ax_cwt_3d.set_ylabel("Frequency [Hz]")
                self.ax_cwt_3d.set_zlabel("Magnitude (dB)" if use_db else "Magnitude")
                ttl = f"3D Scalogram ({getattr(self, 'current_cwt_method', 'CWT')})"
                self.ax_cwt_3d.set_title(ttl)
                try:
                    self.ax_cwt_3d.view_init(elev=elev, azim=azim)
                except Exception:
                    pass
            except Exception:
                pass

            # apply 3D dark styling
            try:
                self._style_3d_axis(self.ax_cwt_3d)
            except Exception:
                pass

            # draw
            try:
                self.canvas_cwt_3d.draw_idle()
            except Exception:
                try:
                    # fallback plain draw
                    self.fig_cwt_3d.canvas.draw()
                except Exception:
                    pass
        except Exception as e:
            # Don't raise — GUI should continue; print for debug
            print("Warning: _plot_cwt_3d failed:", e)
            try:
                import traceback as _tb
                _tb.print_exc()
            except Exception:
                pass
            return


    # ---------- STFT plotting (2D + 3D) ----------
    def _plot_stft_tab(self):
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            try:
                if getattr(self, 'ax_stft_2d', None) is not None:
                    self.ax_stft_2d.clear(); self.canvas_stft_2d.draw_idle()
                if getattr(self, 'ax_stft_3d', None) is not None:
                    self.ax_stft_3d.clear(); self.canvas_stft_3d.draw_idle()
            except Exception:
                pass
            return

        S = np.asarray(self.current_scalogram)
        freqs = np.asarray(self.current_freqs)
        times = np.asarray(self.current_times)
        if S.ndim != 2:
            return

        # 2D spectrogram
        try:
            self.ax_stft_2d.clear()
            extent = [times[0], times[-1], freqs[0], freqs[-1]]
            self.ax_stft_2d.imshow(S, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            self.ax_stft_2d.set_xlabel("Time [s]"); self.ax_stft_2d.set_ylabel("Frequency [Hz]")
            self.ax_stft_2d.set_title("STFT Spectrogram")
            self._style_axis_dark(self.ax_stft_2d)
            try:
                self.fig_stft_2d.tight_layout()
            except Exception:
                pass
            self.canvas_stft_2d.draw_idle()
        except Exception:
            pass

        # 3D surface plot (downsample for performance)
        try:
            self.ax_stft_3d.clear()
            max_cols = 200
            max_rows = 200
            row_step = max(1, int(np.ceil(S.shape[0] / float(max_rows))))
            col_step = max(1, int(np.ceil(S.shape[1] / float(max_cols))))
            Sf = S[::row_step, ::col_step]
            ff = freqs[::row_step]
            tt = times[::col_step]
            TT, FF = np.meshgrid(tt, ff)
            from matplotlib import cm
            try:
                surf = self.ax_stft_3d.plot_surface(TT, FF, Sf, cmap=cm.viridis, linewidth=0, antialiased=False)
                self.ax_stft_3d.set_xlabel("Time [s]"); self.ax_stft_3d.set_ylabel("Frequency [Hz]"); self.ax_stft_3d.set_zlabel("Magnitude")
                try:
                    self.fig_stft_3d.colorbar(surf, ax=self.ax_stft_3d, shrink=0.6)
                except Exception:
                    pass
            except Exception:
                # fallback to wireframe / imshow on 2D axis
                try:
                    self.ax_stft_3d.plot_wireframe(TT, FF, Sf)
                except Exception:
                    pass
            try: self.fig_stft_3d.tight_layout()
            except Exception: pass
            self.canvas_stft_3d.draw_idle()
        except Exception:
            pass

    # ---------- Threshold & CoG plotting ----------
    def _get_thr_params(self):
        ma = int(self.thr_min_area_spin.value())
        if ma == 0:
            min_area = None
        else:
            min_area = ma
        keep_top = int(self.thr_keep_top_spin.value())
        return min_area, keep_top

    def _apply_time_window_filter(self, mask, times, time_window):
        if mask is None:
            return mask
        mask = np.asarray(mask, dtype=bool)
        if times is None or time_window is None:
            return mask
        t0, t1 = float(time_window[0]), float(time_window[1])
        if sp_ndimage is not None:
            labeled, ncomp = sp_ndimage.label(mask)
            if ncomp == 0:
                return np.zeros_like(mask, dtype=bool)
            keep_mask = np.zeros_like(mask, dtype=bool)
            for comp in range(1, ncomp+1):
                comp_mask = (labeled == comp)
                if not comp_mask.any():
                    continue
                try:
                    centroid = sp_ndimage.center_of_mass(comp_mask)
                    _, colc = centroid
                except Exception:
                    ys, xs = np.nonzero(comp_mask)
                    colc = float(xs.mean()) if xs.size else 0.0
                col_idx = int(round(colc))
                col_idx = max(0, min(col_idx, len(times)-1))
                t_comp = float(times[col_idx])
                if (t_comp >= t0) and (t_comp <= t1):
                    keep_mask |= comp_mask
            return keep_mask
        else:
            times_arr = np.asarray(times, dtype=float)
            col_mask_time = (times_arr >= t0) & (times_arr <= t1)
            filtered = np.zeros_like(mask, dtype=bool)
            if col_mask_time.any():
                filtered[:, col_mask_time] = mask[:, col_mask_time]
            return filtered

    def _update_masks_and_cogs(self):
        if self.current_scalogram is None:
            return
        s1 = self.thr_s1.value() / 100.0
        s2 = self.thr_s2.value() / 100.0
        min_area, keep_top = self._get_thr_params()
        try:
            mask1 = threshold_mask(self.current_scalogram, s1, min_area=min_area, keep_top=keep_top, freqs=self.current_freqs, times=self.current_times, time_window=None)
            mask2 = threshold_mask(self.current_scalogram, s2, min_area=min_area, keep_top=keep_top, freqs=self.current_freqs, times=self.current_times, time_window=None)

            if mask1 is not None and mask1.shape != self.current_scalogram.shape:
                if mask1.T.shape == self.current_scalogram.shape:
                    mask1 = mask1.T
            if mask2 is not None and mask2.shape != self.current_scalogram.shape:
                if mask2.T.shape == self.current_scalogram.shape:
                    mask2 = mask2.T

            start_sample, end_sample = self.segment_bounds
            seg_dur_s = float(end_sample - start_sample) / float(self.fs) if (end_sample > start_sample) else (len(self.current_segment)/float(self.fs) if self.current_segment is not None else 0.0)
            if seg_dur_s is None or seg_dur_s <= 0:
                seg_dur_s = float(self.current_times[-1] - self.current_times[0]) if (self.current_times is not None and len(self.current_times)>1) else 0.0
            seg_start_s = float(start_sample) / float(self.fs)
            s1_win = (seg_start_s, seg_start_s + 0.35 * seg_dur_s)
            s2_win = (seg_start_s + 0.35 * seg_dur_s, seg_start_s + 0.95 * seg_dur_s)

            try:
                if self.current_times is not None and len(self.current_times) > 0:
                    mask1 = self._apply_time_window_filter(mask1, self.current_times, s1_win)
                    mask2 = self._apply_time_window_filter(mask2, self.current_times, s2_win)
            except Exception:
                pass

            try:
                cog1 = compute_cog(self.current_scalogram, self.current_freqs, self.current_times, mask=mask1)
            except TypeError:
                cog1 = compute_cog(self.current_scalogram * mask1, self.current_freqs, self.current_times)
            try:
                cog2 = compute_cog(self.current_scalogram, self.current_freqs, self.current_times, mask=mask2)
            except TypeError:
                cog2 = compute_cog(self.current_scalogram * mask2, self.current_freqs, self.current_times)
        except Exception:
            mask1 = np.zeros_like(self.current_scalogram, dtype=bool)
            mask2 = np.zeros_like(self.current_scalogram, dtype=bool)
            cog1 = None; cog2 = None

        self.current_masks = {'S1': mask1, 'S2': mask2}
        self.current_cogs = {'S1': cog1, 'S2': cog2}

    def _plot_threshold_tab(self):
        self.ax_thr.cla()
        try:
            for ax in list(self.fig_thr.axes):
                if ax is not self.ax_thr:
                    self.fig_thr.delaxes(ax)
        except Exception:
            pass
        if getattr(self, 'thr_cb', None) is not None:
            try: self.thr_cb.remove()
            except Exception: pass
            self.thr_cb = None
        if getattr(self, 'thr_cax', None) is not None:
            try: self.fig_thr.delaxes(self.thr_cax)
            except Exception: pass
            self.thr_cax = None

        try:
            for art in getattr(self, 'last_thr_scat', []):
                try: art.remove()
                except Exception: pass
        except Exception:
            pass
        self.last_thr_scat = []
        self.last_thr_contours = []

        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            self.canvas_thr.draw_idle()
            return

        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(self.current_scalogram, self.current_freqs, self.current_times)
        except Exception:
            scal = self.current_scalogram
            freqs = self.current_freqs
            times = self.current_times

        self.current_scalogram = scal; self.current_freqs = freqs; self.current_times = times
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        
        # update 2D CWT then 3D
        self._plot_cwt_tab()
        self._plot_cwt_3d()

        self.ax_thr.clear()
        self.thr_im = self.ax_thr.imshow(scal, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
        try:
            divider = make_axes_locatable(self.ax_thr)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            self.thr_cb = self.fig_thr.colorbar(self.thr_im, cax=cax)
            self.thr_cax = cax
        except Exception:
            self.thr_cb = None
            self.thr_cax = None

        mask1 = self.current_masks.get('S1', None)
        mask2 = self.current_masks.get('S2', None)

        def safe_contour(ax, times_arr, freqs_arr, mask_arr, **kwargs):
            if mask_arr is None:
                return None
            mask_arr = np.asarray(mask_arr)
            if mask_arr.shape != (len(freqs_arr), len(times_arr)):
                if mask_arr.T.shape == (len(freqs_arr), len(times_arr)):
                    mask_arr = mask_arr.T
                else:
                    return None
            try:
                c = ax.contour(times_arr, freqs_arr, mask_arr.astype(int), levels=[0.5], **kwargs)
                return c
            except Exception:
                return None

        if mask1 is not None and mask1.any():
            c1 = safe_contour(self.ax_thr, times, freqs, mask1, colors='white', linewidths=1)
            if c1 is not None:
                self.last_thr_contours.append(c1)
        if mask2 is not None and mask2.any():
            c2 = safe_contour(self.ax_thr, times, freqs, mask2, colors='cyan', linewidths=1)
            if c2 is not None:
                self.last_thr_contours.append(c2)

        cog1 = self.current_cogs.get('S1', None)
        cog2 = self.current_cogs.get('S2', None)
        if cog1 is not None:
            try:
                art1 = self.ax_thr.scatter([cog1[0]], [cog1[1]], marker='x', s=80, c='white', zorder=5, label='S1 CoG')
                self.last_thr_scat.append(art1)
            except Exception:
                pass
        if cog2 is not None:
            try:
                art2 = self.ax_thr.scatter([cog2[0]], [cog2[1]], marker='o', s=80, edgecolors='cyan', facecolors='none', zorder=5, label='S2 CoG')
                self.last_thr_scat.append(art2)
            except Exception:
                pass

        try:
            self.ax_thr.set_xlim(times[0], times[-1])
            self.ax_thr.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass

        self.ax_thr.set_xlabel("Time [s]"); self.ax_thr.set_ylabel("Frequency [Hz]")
        try:
            proxies = []
            labels = []
            proxies.append(Line2D([0], [0], marker='x', color='w', linestyle=''))
            labels.append('S1 CoG')
            proxies.append(Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='cyan', color='w', linestyle=''))
            labels.append('S2 CoG')
            self.ax_thr.legend(proxies, labels, loc='upper right', fontsize='small', framealpha=0.6)
        except Exception:
            pass

        self._style_axis_dark(self.ax_thr)
        try: self.fig_thr.tight_layout()
        except Exception: pass
        self.canvas_thr.draw_idle()

    # ---------- handlers ----------
    def _on_cwt_params_changed(self):
        if self.current_segment is None:
            return
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            self.status.showMessage("Invalid CWT parameters")
            return
        self._compute_cwt_for_current_segment(fmin, fmax, nfreqs)

    def _on_cwt_backend_changed(self, _idx):
        # reused as a generic TF-change handler (including TF method combo)
        if self.current_segment is None:
            return
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            fmin, fmax, nfreqs = DEFAULT_CWT_FMIN, DEFAULT_CWT_FMAX, DEFAULT_CWT_NFREQS
        self._compute_cwt_for_current_segment(fmin, fmax, nfreqs)

    def _compute_cwt_for_current_segment(self, fmin, fmax, nfreqs):
        # This function remains primarily for CWT but respects TF method selection
        if self.current_segment is None:
            return

        tf_choice = 'cwt'
        try:
            tf_choice = str(self.tf_method_combo.currentText()).strip().lower()
        except Exception:
            tf_choice = 'cwt'

        if tf_choice == 'stft':
            # compute STFT for current segment
            nperseg = int(self.stft_nperseg.value()) if hasattr(self, 'stft_nperseg') else None
            noverlap = int(self.stft_noverlap.value()) if hasattr(self, 'stft_noverlap') else None
            nfft = int(self.stft_nfft.value()) if hasattr(self, 'stft_nfft') else None
            if compute_stft is None:
                self.status.showMessage("STFT module not available (STFT_PCG.py missing).")
                return
            try:
                scalogram, freqs, times_rel, method = compute_stft(self.current_segment, fs=self.fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann', mode='magnitude')
            except Exception as e:
                self.status.showMessage(f"STFT error: {e}")
                print(traceback.format_exc())
                return
        else:
            backend_sel = str(self.cwt_backend_combo.currentText()).strip().lower()
            compute_kwargs = {}
            if backend_sel == 'pascal':
                compute_kwargs['a0'] = float(self.cwt_a0_spin.value())
                compute_kwargs['a_step'] = float(self.cwt_astep_spin.value())
                compute_kwargs['col_count'] = int(self.cwt_colcount_spin.value())
                if self.cwt_use_freqs_target.isChecked():
                    compute_kwargs['freqs_target'] = np.linspace(fmin, fmax, nfreqs)
            try:
                scalogram, freqs, times_rel, method = compute_cwt(self.current_segment, fs=self.fs, fmin=fmin, fmax=fmax, n_freqs=nfreqs, backend=backend_sel, **compute_kwargs)
            except Exception as e:
                self.status.showMessage(f"CWT error: {e}")
                print(traceback.format_exc())
                return

        try:
            seg_dur = float(len(self.current_segment)) / float(self.fs) if (self.current_segment is not None) else None
            if times_rel is not None:
                times_rel = np.asarray(times_rel, dtype=float)
                if seg_dur is not None and times_rel.size > 0:
                    if times_rel.max() > (seg_dur * 1.2):
                        times_rel = times_rel / float(self.fs)
        except Exception:
            pass

        start_sample = int(self.segment_bounds[0]) if hasattr(self, 'segment_bounds') else 0
        times_abs = times_rel + start_sample / float(self.fs) if (times_rel is not None and len(times_rel) > 0) else None

        try:
            scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(scalogram, freqs, times_abs)
        except Exception:
            self.status.showMessage("TF returned invalid scalogram shape")
            return

        # set result
        self.current_scalogram = scal2
        self.current_freqs = freqs2
        self.current_times = times2
        self.current_cwt_method = method
        self.method_label.setText(f"TF method: {method} (mode: {tf_choice})")
        
        # update 2D CWT then 3D
        self._plot_cwt_tab()
        self._plot_cwt_3d()

        self.current_masks = {}
        self.current_cogs = {}

        self._update_masks_and_cogs()
        # plot proper tab
        if tf_choice == 'stft':
            self._plot_stft_tab()
        else:
            self._plot_cwt_tab()
        self._plot_threshold_tab()
        self.status.showMessage(f"TF recomputed ({method}) using mode={tf_choice}")

    def _on_thr_slider_changed(self, _val):
        self.thr_s1_label.setText(f"{self.thr_s1.value()/100.0:.2f}")
        self.thr_s2_label.setText(f"{self.thr_s2.value()/100.0:.2f}")
        self._update_masks_and_cogs()
        self._plot_threshold_tab()

    def _on_thr_params_changed(self, _v=None):
        if self.current_scalogram is None:
            return
        self._update_masks_and_cogs()
        self._plot_threshold_tab()

    def _on_save_results(self):
        if self.current_scalogram is None or self.current_segment is None:
            self.status.showMessage("No result to save")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder to save results", os.getcwd())
        if not folder:
            return
        base = (self.record_label.text().split()[1] if "Loaded:" in self.record_label.text() else "record")
        beat_idx = int(self.seg_beat_slider.value()) if self.seg_beat_slider.isEnabled() else 0
        s1 = self.thr_s1.value() / 100.0; s2 = self.thr_s2.value() / 100.0
        min_area, keep_top = self._get_thr_params()
        filename_base = f"{base}_beat{beat_idx}_s1_{int(s1*100)}_s2_{int(s2*100)}"
        png_path = os.path.join(folder, filename_base + ".png")
        json_path = os.path.join(folder, filename_base + ".json")
        try:
            self.fig_thr.savefig(png_path, dpi=300, bbox_inches='tight')
            metadata = {
                'record': base,
                'fs': int(self.fs),
                'pcg_channel': int(self.pcg_idx),
                'ecg_channel': int(self.ecg_idx) if self.ecg_idx is not None else None,
                'beat_index': beat_idx,
                'segment_samples': {'start': int(self.segment_bounds[0]), 'end': int(self.segment_bounds[1])},
                'cwt_method': str(self.current_cwt_method),
                'cwt_params': {
                    'fmin': float(self.cwt_fmin.text()), 'fmax': float(self.cwt_fmax.text()), 'n_freqs': int(self.cwt_nfreqs.text()),
                    'backend': str(self.cwt_backend_combo.currentText()),
                    'use_explicit_freqs': bool(self.cwt_use_freqs_target.isChecked()),
                    'col_count': int(self.cwt_colcount_spin.value()),
                    'a0': float(self.cwt_a0_spin.value()), 'a_step': float(self.cwt_astep_spin.value())
                },
                'thresholds': {'S1': s1, 'S2': s2, 'min_area': min_area, 'keep_top': keep_top},
                'CoG': {
                    'S1': None if self.current_cogs.get('S1') is None else {'t_s': float(self.current_cogs['S1'][0]), 'f_hz': float(self.current_cogs['S1'][1])},
                    'S2': None if self.current_cogs.get('S2') is None else {'t_s': float(self.current_cogs['S2'][0]), 'f_hz': float(self.current_cogs['S2'][1])},
                }
            }
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.status.showMessage(f"Saved PNG: {png_path} and JSON: {json_path}")
        except Exception as e:
            self.status.showMessage(f"Save failed: {e}")
            print(traceback.format_exc())

    # ---------------- responsiveness: keep canvas sizes up-to-date ----------------
    def resizeEvent(self, event):
        # update canvas minimum sizes when window resized so scroll areas recompute properly
        try:
            self._update_all_canvas_min_sizes()
        except Exception:
            pass
        super().resizeEvent(event)

    def _update_all_canvas_min_sizes(self):
        # compute new minimum sizes from figsize * dpi and re-apply
        canvases = [
            getattr(self, 'canvas_pt_raw', None),
            getattr(self, 'canvas_pt_proc', None),
            getattr(self, 'canvas_pt_pcg', None),
            getattr(self, 'canvas_seg_ecg', None),
            getattr(self, 'canvas_seg_pcg', None),
            getattr(self, 'canvas_cwt', None),
            getattr(self, 'canvas_thr', None),
            getattr(self, 'canvas_stft_2d', None),
            getattr(self, 'canvas_stft_3d', None),
        ]
        for c in canvases:
            if c is None:
                continue
            try:
                fig = getattr(c, 'figure', None)
                if fig is None:
                    fig = getattr(c, 'figure', None)
                if fig is not None:
                    dpi = fig.get_dpi()
                    w_px = int(max(300, fig.get_figwidth() * dpi))
                    h_px = int(max(160, fig.get_figheight() * dpi))
                    c.setMinimumWidth(w_px)
                    c.setMinimumHeight(h_px)
                    c.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
                    c.updateGeometry()
                    try:
                        c.draw_idle()
                    except Exception:
                        pass
            except Exception:
                pass

def run_app():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = PCGAnalyzerGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
