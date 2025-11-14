# GUI.py

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
from matplotlib.colors import Normalize

# 3D plotting (import for side-effects)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Try to import optional modules used by GUI algorithms
try:
    from scipy import ndimage as sp_ndimage
except Exception:
    sp_ndimage = None

# Local modules (should exist in your project)
try:
    from utils_io import try_load_record
except Exception:
    def try_load_record(base):
        raise RuntimeError("utils_io.try_load_record not found")

try:
    from Pan_Tompkins import detect_r_peaks_with_fallback, plot_pt_pipeline
except Exception:
    def detect_r_peaks_with_fallback(*args, **kwargs):
        raise RuntimeError("Pan_Tompkins.detect_r_peaks_with_fallback not found")
    def plot_pt_pipeline(*args, **kwargs):
        raise RuntimeError("Pan_Tompkins.plot_pt_pipeline not found")

try:
    from Segmentation_ECG_to_PCG import segment_one_cycle, choose_clean_beat
except Exception:
    def segment_one_cycle(*args, **kwargs):
        raise RuntimeError("Segmentation_ECG_to_PCG.segment_one_cycle not found")
    def choose_clean_beat(*args, **kwargs):
        return 0

try:
    from CWT_PCG import compute_cwt
except Exception:
    def compute_cwt(*args, **kwargs):
        raise RuntimeError("CWT_PCG.compute_cwt not found")

try:
    from Threshold_Plot_CoG import threshold_mask, compute_cog
except Exception:
    def threshold_mask(*args, **kwargs):
        return None
    def compute_cog(*args, **kwargs):
        return None

# STFT module (we expect a compute_stft in STFT_PCG.py)
try:
    from STFT_PCG import compute_stft
except Exception:
    compute_stft = None

# Defaults
DEFAULT_FS = 2000
DEFAULT_CWT_FMIN = 20.0
DEFAULT_CWT_FMAX = 200.0
DEFAULT_CWT_NFREQS = 120
DEFAULT_CWT_COLCOUNT = 300
DEFAULT_PASCAL_A0 = 0.0019
DEFAULT_PASCAL_A_STEP = 0.00031

# UI theme
QT_STYLE = """
QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #071224, stop:1 #0b3b70); color: #E6F0FA; font-family: "Segoe UI", Roboto, Arial; font-size: 11px; }
QTabWidget::pane { border: 1px solid #0b345f; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #062033, stop:1 #0a4b86); padding: 6px; }
QTabBar::tab { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #08304a, stop:1 #0b4f80); border: 1px solid #08304a; padding: 8px; margin: 2px; min-width: 120px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0b66a6, stop:1 #0380ff); color: white; font-weight: bold; }
QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #09547a, stop:1 #0b73b3); color: #F3FBFF; border-radius:6px; padding:6px 10px; }
QPushButton:disabled { background:#1b2f44; color:#7f97b0; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: rgba(255,255,255,0.04); color:#eaf6ff; border-radius:4px; padding:4px; }
QLabel { color: #dbeeff; }
QSlider::groove:horizontal { background: rgba(255,255,255,0.06); height:8px; border-radius:4px; }
QSlider::handle:horizontal { background: #0b73b3; width: 14px; border-radius:7px; margin:-3px 0; }
QCheckBox { color:#eaf6ff; }
"""

FIG_BG = "#071224"
AX_BG = "#071224"

class PCGAnalyzerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCG/ECG Analyzer - Jeremia Christ Immanuel Manalu (5023231017)")
        self.resize(1250, 920)
        self.setStyleSheet(QT_STYLE)

        # state
        self.record_p_signal = None
        self.fs = DEFAULT_FS
        self.sig_names = []
        self.ecg_idx = None
        self.pcg_idx = None
        self.r_peaks = np.array([], dtype=int)

        # caches / results
        self.current_segment = None
        self.segment_bounds = (0, 0)
        self.current_scalogram = None
        self.current_freqs = None
        self.current_times = None
        self.current_cwt_method = None
        self.current_masks = {}
        self.current_cogs = {}

        # plotting handles
        self.cwt_im = None; self.cwt_cb = None; self.cwt_cax = None
        self.thr_im = None; self.thr_cb = None; self.thr_cax = None
        self.cwt_3d_surf = None

        # STFT handles
        self.stft_cb = None

        # Build UI
        self._build_ui()
        self._connect_signals()

    # ---------------- UI helpers ----------------
    def _mk_figure_canvas(self, figsize=(10, 4.5), dpi=100):
        fig = Figure(figsize=figsize, dpi=dpi, facecolor=FIG_BG)
        canvas = FigureCanvas(fig)
        return fig, canvas

    def _style_axis_dark(self, ax):
        try:
            ax.set_facecolor(AX_BG)
            for spine in ax.spines.values():
                spine.set_color('#6e9fbf')
            ax.tick_params(colors='#cfe8ff', which='both')
            ax.xaxis.label.set_color('#dbeeff'); ax.yaxis.label.set_color('#dbeeff')
            ax.title.set_color('#eaf6ff')
            ax.grid(True, linestyle='--', linewidth=0.4, color=(0.08,0.12,0.2))
        except Exception:
            pass

    def _style_3d_axis(self, ax):
        try:
            # Attempt to set pane colors (may fail on some backends)
            try:
                ax.w_xaxis.set_pane_color((0.03,0.07,0.14,1.0))
                ax.w_yaxis.set_pane_color((0.03,0.07,0.14,1.0))
                ax.w_zaxis.set_pane_color((0.03,0.07,0.14,1.0))
            except Exception:
                try: ax.set_facecolor((0.03,0.07,0.14))
                except Exception: pass
            # Tick/label colors
            try:
                for lbl in (ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels() + ax.zaxis.get_ticklabels()):
                    lbl.set_color('#DDEFF8')
                ax.xaxis.label.set_color('#DDEFF8'); ax.yaxis.label.set_color('#DDEFF8'); ax.zaxis.label.set_color('#DDEFF8')
                ax.title.set_color('#EAF6FF')
                ax.grid(True, linestyle=':', color=(0.08,0.12,0.2), linewidth=0.5)
            except Exception:
                pass
        except Exception:
            pass

    def _make_canvas_widget(self, canvas, toolbar=None, fixed=True, pad=(4,4,4,4)):
        w = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(w)
        wl.setContentsMargins(*pad)
        wl.setSpacing(4)
        if toolbar is not None:
            wl.addWidget(toolbar)
        try:
            fig = getattr(canvas, 'figure', None)
            if fig is not None and fixed:
                dpi = fig.get_dpi()
                w_px = int(max(480, fig.get_figwidth() * dpi))
                h_px = int(max(160, fig.get_figheight() * dpi))
                canvas.setFixedWidth(w_px)
                canvas.setFixedHeight(h_px)
                canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            else:
                canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        except Exception:
            pass
        wl.addWidget(canvas)
        return w

    # ---------------- Build UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_v = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # top controls
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

        # --- Pan-Tompkins tab ---
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

        # PT canvases
        self.fig_pt_raw, self.canvas_pt_raw = self._mk_figure_canvas(figsize=(10, 2.8))
        self.toolbar_pt_raw = NavigationToolbar(self.canvas_pt_raw, self)
        self.ax_pt_ecg_raw = self.fig_pt_raw.add_subplot(111)
        self._style_axis_dark(self.ax_pt_ecg_raw)

        self.fig_pt_proc, self.canvas_pt_proc = self._mk_figure_canvas(figsize=(10, 2.4))
        self.toolbar_pt_proc = NavigationToolbar(self.canvas_pt_proc, self)
        self.ax_pt_ecg_proc = self.fig_pt_proc.add_subplot(111)
        self._style_axis_dark(self.ax_pt_ecg_proc)

        self.fig_pt_pcg, self.canvas_pt_pcg = self._mk_figure_canvas(figsize=(10, 2.8))
        self.toolbar_pt_pcg = NavigationToolbar(self.canvas_pt_pcg, self)
        self.ax_pt_pcg = self.fig_pt_pcg.add_subplot(111)
        self._style_axis_dark(self.ax_pt_pcg)

        container_pt = QtWidgets.QWidget()
        v_pt = QtWidgets.QVBoxLayout(container_pt)
        v_pt.setContentsMargins(2,2,2,2)
        v_pt.setSpacing(8)
        v_pt.addWidget(self._make_canvas_widget(self.canvas_pt_raw, self.toolbar_pt_raw, fixed=True))
        v_pt.addWidget(self._make_canvas_widget(self.canvas_pt_proc, self.toolbar_pt_proc, fixed=True))
        v_pt.addWidget(self._make_canvas_widget(self.canvas_pt_pcg, self.toolbar_pt_pcg, fixed=True))
        v_pt.addStretch()

        scroll_pt = QtWidgets.QScrollArea()
        scroll_pt.setWidgetResizable(True)
        scroll_pt.setWidget(container_pt)
        t1.addWidget(scroll_pt, stretch=1)
        self.tabs.addTab(self.tab_pt, "Pan-Tompkins")

        # --- Segmentation tab ---
        self.tab_seg = QtWidgets.QWidget(); t2 = QtWidgets.QVBoxLayout(self.tab_seg)
        seg_ctrl = QtWidgets.QHBoxLayout()
        seg_ctrl.addWidget(QtWidgets.QLabel("Beat index:"))
        self.seg_beat_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seg_beat_slider.setEnabled(False)
        self.seg_beat_label = QtWidgets.QLabel("0")
        seg_ctrl.addWidget(self.seg_beat_slider); seg_ctrl.addWidget(self.seg_beat_label)
        seg_ctrl.addStretch(); t2.addLayout(seg_ctrl)

        self.fig_seg_ecg, self.canvas_seg_ecg = self._mk_figure_canvas(figsize=(10, 2.6))
        self.toolbar_seg_ecg = NavigationToolbar(self.canvas_seg_ecg, self)
        self.ax_seg_ecg = self.fig_seg_ecg.add_subplot(111)
        self._style_axis_dark(self.ax_seg_ecg)

        self.fig_seg_pcg, self.canvas_seg_pcg = self._mk_figure_canvas(figsize=(10, 2.6))
        self.toolbar_seg_pcg = NavigationToolbar(self.canvas_seg_pcg, self)
        self.ax_seg_pcg = self.fig_seg_pcg.add_subplot(111)
        self._style_axis_dark(self.ax_seg_pcg)

        container_seg = QtWidgets.QWidget()
        v_seg = QtWidgets.QVBoxLayout(container_seg)
        v_seg.setContentsMargins(2,2,2,2); v_seg.setSpacing(8)
        v_seg.addWidget(self._make_canvas_widget(self.canvas_seg_ecg, self.toolbar_seg_ecg, fixed=True))
        v_seg.addWidget(self._make_canvas_widget(self.canvas_seg_pcg, self.toolbar_seg_pcg, fixed=True))
        v_seg.addStretch()
        scroll_seg = QtWidgets.QScrollArea(); scroll_seg.setWidgetResizable(True); scroll_seg.setWidget(container_seg)
        t2.addWidget(scroll_seg, stretch=1)
        self.tabs.addTab(self.tab_seg, "Segmentation")

        # --- CWT tab (2D + 3D) ---
        self.tab_cwt = QtWidgets.QWidget(); t3 = QtWidgets.QVBoxLayout(self.tab_cwt)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("fmin")); self.cwt_fmin = QtWidgets.QLineEdit(str(DEFAULT_CWT_FMIN)); self.cwt_fmin.setMaximumWidth(90); ctrl.addWidget(self.cwt_fmin)
        ctrl.addWidget(QtWidgets.QLabel("fmax")); self.cwt_fmax = QtWidgets.QLineEdit(str(DEFAULT_CWT_FMAX)); self.cwt_fmax.setMaximumWidth(90); ctrl.addWidget(self.cwt_fmax)
        ctrl.addWidget(QtWidgets.QLabel("n_freqs")); self.cwt_nfreqs = QtWidgets.QLineEdit(str(DEFAULT_CWT_NFREQS)); self.cwt_nfreqs.setMaximumWidth(90); ctrl.addWidget(self.cwt_nfreqs)
        ctrl.addWidget(QtWidgets.QLabel("Backend")); self.cwt_backend_combo = QtWidgets.QComboBox(); self.cwt_backend_combo.setMaximumWidth(150)
        self.cwt_backend_combo.addItems(["pascal", "pywt", "scipy", "spectrogram"]); self.cwt_backend_combo.setCurrentText("pascal"); ctrl.addWidget(self.cwt_backend_combo)
        ctrl.addWidget(QtWidgets.QLabel("TF method")); self.tf_method_combo = QtWidgets.QComboBox(); self.tf_method_combo.setMaximumWidth(140)
        self.tf_method_combo.addItems(["CWT", "STFT"]); self.tf_method_combo.setCurrentText("CWT"); ctrl.addWidget(self.tf_method_combo)
        self.cwt_use_freqs_target = QtWidgets.QCheckBox("Use explicit freqs (linspace fmin->fmax)"); self.cwt_use_freqs_target.setChecked(True); ctrl.addWidget(self.cwt_use_freqs_target)
        ctrl.addWidget(QtWidgets.QLabel("col_count")); self.cwt_colcount_spin = QtWidgets.QSpinBox(); self.cwt_colcount_spin.setRange(16,2000); self.cwt_colcount_spin.setValue(DEFAULT_CWT_COLCOUNT); self.cwt_colcount_spin.setMaximumWidth(100); ctrl.addWidget(self.cwt_colcount_spin)
        ctrl.addWidget(QtWidgets.QLabel("a0")); self.cwt_a0_spin = QtWidgets.QDoubleSpinBox(); self.cwt_a0_spin.setDecimals(7); self.cwt_a0_spin.setRange(0.000001,0.01); self.cwt_a0_spin.setSingleStep(0.000001); self.cwt_a0_spin.setValue(DEFAULT_PASCAL_A0); self.cwt_a0_spin.setMaximumWidth(120); ctrl.addWidget(self.cwt_a0_spin)
        ctrl.addWidget(QtWidgets.QLabel("a_step")); self.cwt_astep_spin = QtWidgets.QDoubleSpinBox(); self.cwt_astep_spin.setDecimals(8); self.cwt_astep_spin.setRange(0.0000001,0.001); self.cwt_astep_spin.setSingleStep(0.0000001); self.cwt_astep_spin.setValue(DEFAULT_PASCAL_A_STEP); self.cwt_astep_spin.setMaximumWidth(120); ctrl.addWidget(self.cwt_astep_spin)
        ctrl.addStretch()
        self.method_label = QtWidgets.QLabel("CWT method: (not computed)")
        ctrl.addWidget(self.method_label)
        t3.addLayout(ctrl)
        
        # --- CWT 3D controls (toggle dB + preset view buttons) ---
        cwt_3d_layout = QtWidgets.QHBoxLayout()
        self.cwt_3d_db_cb = QtWidgets.QCheckBox("3D use dB")
        self.cwt_3d_db_cb.setChecked(True)
        self.cwt_3d_db_cb.setToolTip("Toggle dB scaling for the 3D CWT surface")
        cwt_3d_layout.addWidget(self.cwt_3d_db_cb)
        cwt_3d_layout.addWidget(QtWidgets.QLabel("3D Views:"))
        self.cwt_view_oblique = QtWidgets.QPushButton("Oblique")
        self.cwt_view_iso = QtWidgets.QPushButton("Iso")
        self.cwt_view_top = QtWidgets.QPushButton("Top")
        for b in (self.cwt_view_oblique, self.cwt_view_iso, self.cwt_view_top):
            b.setMaximumWidth(80)
            cwt_3d_layout.addWidget(b)
        cwt_3d_layout.addStretch()
        t3.addLayout(cwt_3d_layout)


        self.fig_cwt, self.canvas_cwt = self._mk_figure_canvas(figsize=(10, 4.2))
        self.toolbar_cwt = NavigationToolbar(self.canvas_cwt, self)
        self.ax_cwt = self.fig_cwt.add_subplot(111)
        self._style_axis_dark(self.ax_cwt)

        self.fig_cwt_3d, self.canvas_cwt_3d = self._mk_figure_canvas(figsize=(10, 4.0))
        self.toolbar_cwt_3d = NavigationToolbar(self.canvas_cwt_3d, self)
        try:
            self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111, projection='3d')
        except Exception:
            self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111)
        self._style_3d_axis(self.ax_cwt_3d)

        container_cwt = QtWidgets.QWidget()
        v_cwt = QtWidgets.QVBoxLayout(container_cwt)
        v_cwt.setContentsMargins(2,2,2,2); v_cwt.setSpacing(8)
        v_cwt.addWidget(self._make_canvas_widget(self.canvas_cwt, self.toolbar_cwt, fixed=True))
        v_cwt.addWidget(self._make_canvas_widget(self.canvas_cwt_3d, self.toolbar_cwt_3d, fixed=True))
        v_cwt.addStretch()
        scroll_cwt = QtWidgets.QScrollArea(); scroll_cwt.setWidgetResizable(True); scroll_cwt.setWidget(container_cwt)
        t3.addWidget(scroll_cwt, stretch=1)
        self.tabs.addTab(self.tab_cwt, "CWT")

        # --- STFT tab ---
        self.tab_stft = QtWidgets.QWidget(); t_stft = QtWidgets.QVBoxLayout(self.tab_stft)
        stft_ctrl = QtWidgets.QHBoxLayout()
        stft_ctrl.addWidget(QtWidgets.QLabel("nperseg"))
        self.stft_nperseg = QtWidgets.QSpinBox(); self.stft_nperseg.setRange(16,8192); self.stft_nperseg.setValue(256); self.stft_nperseg.setMaximumWidth(100)
        stft_ctrl.addWidget(self.stft_nperseg)
        stft_ctrl.addWidget(QtWidgets.QLabel("noverlap"))
        self.stft_noverlap = QtWidgets.QSpinBox(); self.stft_noverlap.setRange(0,8191); self.stft_noverlap.setValue(128); self.stft_noverlap.setMaximumWidth(100)
        stft_ctrl.addWidget(self.stft_noverlap)
        # --- Add axis limit controls (time/freq) ---
        stft_ctrl.addStretch()
        stft_ctrl.addWidget(QtWidgets.QLabel("Time min:"))
        self.stft_xmin = QtWidgets.QDoubleSpinBox(); self.stft_xmin.setDecimals(4); self.stft_xmin.setRange(-1e6,1e6); self.stft_xmin.setMaximumWidth(110)
        stft_ctrl.addWidget(self.stft_xmin)
        stft_ctrl.addWidget(QtWidgets.QLabel("Time max:"))
        self.stft_xmax = QtWidgets.QDoubleSpinBox(); self.stft_xmax.setDecimals(4); self.stft_xmax.setRange(-1e6,1e6); self.stft_xmax.setMaximumWidth(110)
        stft_ctrl.addWidget(self.stft_xmax)
        stft_ctrl.addWidget(QtWidgets.QLabel("Freq min:"))
        self.stft_ymin = QtWidgets.QDoubleSpinBox(); self.stft_ymin.setDecimals(3); self.stft_ymin.setRange(0.0,1e6); self.stft_ymin.setMaximumWidth(110)
        stft_ctrl.addWidget(self.stft_ymin)
        stft_ctrl.addWidget(QtWidgets.QLabel("Freq max:"))
        self.stft_ymax = QtWidgets.QDoubleSpinBox(); self.stft_ymax.setDecimals(3); self.stft_ymax.setRange(0.0,1e6); self.stft_ymax.setMaximumWidth(110)
        stft_ctrl.addWidget(self.stft_ymax)
        self.stft_apply_btn = QtWidgets.QPushButton("Apply")
        self.stft_auto_btn = QtWidgets.QPushButton("Auto")
        self.stft_apply_btn.setMaximumWidth(80); self.stft_auto_btn.setMaximumWidth(80)
        stft_ctrl.addWidget(self.stft_apply_btn); stft_ctrl.addWidget(self.stft_auto_btn)
        t_stft.addLayout(stft_ctrl)

        # STFT 2D canvas
        self.fig_stft_2d, self.canvas_stft_2d = self._mk_figure_canvas(figsize=(10, 3.8))
        self.toolbar_stft_2d = NavigationToolbar(self.canvas_stft_2d, self)
        self.ax_stft_2d = self.fig_stft_2d.add_subplot(111)
        self._style_axis_dark(self.ax_stft_2d)

        # STFT 3D canvas
        self.fig_stft_3d, self.canvas_stft_3d = self._mk_figure_canvas(figsize=(10, 4.0))
        self.toolbar_stft_3d = NavigationToolbar(self.canvas_stft_3d, self)
        try:
            self.ax_stft_3d = self.fig_stft_3d.add_subplot(111, projection='3d')
        except Exception:
            self.ax_stft_3d = self.fig_stft_3d.add_subplot(111)
        self._style_3d_axis(self.ax_stft_3d)
        
        # --- STFT 3D controls (toggle dB + preset view buttons) ---
        stft_3d_layout = QtWidgets.QHBoxLayout()
        self.stft_3d_db_cb = QtWidgets.QCheckBox("3D use dB")
        self.stft_3d_db_cb.setChecked(True)
        self.stft_3d_db_cb.setToolTip("Toggle dB scaling for the 3D STFT surface")
        stft_3d_layout.addWidget(self.stft_3d_db_cb)
        stft_3d_layout.addWidget(QtWidgets.QLabel("3D Views:"))
        self.stft_view_oblique = QtWidgets.QPushButton("Oblique")
        self.stft_view_iso = QtWidgets.QPushButton("Iso")
        self.stft_view_top = QtWidgets.QPushButton("Top")
        for b in (self.stft_view_oblique, self.stft_view_iso, self.stft_view_top):
            b.setMaximumWidth(80)
            stft_3d_layout.addWidget(b)
        stft_3d_layout.addStretch()
        t_stft.addLayout(stft_3d_layout)


        container_stft = QtWidgets.QWidget()
        v_stft_container = QtWidgets.QVBoxLayout(container_stft)
        v_stft_container.setContentsMargins(2,2,2,2); v_stft_container.setSpacing(8)
        v_stft_container.addWidget(self._make_canvas_widget(self.canvas_stft_2d, self.toolbar_stft_2d, fixed=True))
        v_stft_container.addWidget(self._make_canvas_widget(self.canvas_stft_3d, self.toolbar_stft_3d, fixed=True))
        v_stft_container.addStretch()
        scroll_stft = QtWidgets.QScrollArea(); scroll_stft.setWidgetResizable(True); scroll_stft.setWidget(container_stft)
        t_stft.addWidget(scroll_stft, stretch=1)
        self.tabs.addTab(self.tab_stft, "STFT")

        # --- Threshold & CoG tab ---
        self.tab_thr = QtWidgets.QWidget(); t4 = QtWidgets.QVBoxLayout(self.tab_thr)
        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel("S1 thr")); self.thr_s1 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s1.setMinimum(1); self.thr_s1.setMaximum(99); self.thr_s1.setValue(60); self.thr_s1.setEnabled(False)
        thr_row.addWidget(self.thr_s1); self.thr_s1_label = QtWidgets.QLabel("0.60"); thr_row.addWidget(self.thr_s1_label)
        thr_row.addWidget(QtWidgets.QLabel("S2 thr")); self.thr_s2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s2.setMinimum(1); self.thr_s2.setMaximum(99); self.thr_s2.setValue(10); self.thr_s2.setEnabled(False)
        thr_row.addWidget(self.thr_s2); self.thr_s2_label = QtWidgets.QLabel("0.10"); thr_row.addWidget(self.thr_s2_label)
        thr_row.addStretch()
        thr_row.addWidget(QtWidgets.QLabel("min_area")); self.thr_min_area_spin = QtWidgets.QSpinBox(); self.thr_min_area_spin.setRange(0,1000); self.thr_min_area_spin.setValue(0); self.thr_min_area_spin.setMaximumWidth(110)
        thr_row.addWidget(self.thr_min_area_spin)
        thr_row.addWidget(QtWidgets.QLabel("keep_top")); self.thr_keep_top_spin = QtWidgets.QSpinBox(); self.thr_keep_top_spin.setRange(1,20); self.thr_keep_top_spin.setValue(3); self.thr_keep_top_spin.setMaximumWidth(80)
        thr_row.addWidget(self.thr_keep_top_spin)
        t4.addLayout(thr_row)

        self.fig_thr, self.canvas_thr = self._mk_figure_canvas(figsize=(10, 4.5))
        self.toolbar_thr = NavigationToolbar(self.canvas_thr, self)
        self.ax_thr = self.fig_thr.add_subplot(111)
        self._style_axis_dark(self.ax_thr)

        container_thr = QtWidgets.QWidget()
        v_thr = QtWidgets.QVBoxLayout(container_thr)
        v_thr.setContentsMargins(2,2,2,2); v_thr.setSpacing(8)
        v_thr.addWidget(self._make_canvas_widget(self.canvas_thr, self.toolbar_thr, fixed=True))
        v_thr.addStretch()
        scroll_thr = QtWidgets.QScrollArea(); scroll_thr.setWidgetResizable(True); scroll_thr.setWidget(container_thr)
        t4.addWidget(scroll_thr, stretch=1)
        self.tabs.addTab(self.tab_thr, "Threshold & CoG")

        # status
        self.status = self.statusBar()
        self.status.showMessage("Ready")

    # ---------------- connections ----------------
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
        self.cwt_backend_combo.currentIndexChanged.connect(self._on_cwt_params_changed)
        self.cwt_colcount_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_a0_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_astep_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.tf_method_combo.currentIndexChanged.connect(self._on_cwt_params_changed)
        self.stft_nperseg.valueChanged.connect(self._on_stft_params_changed)
        self.stft_noverlap.valueChanged.connect(self._on_stft_params_changed)
        self.stft_apply_btn.clicked.connect(self._apply_stft_axis_limits)
        self.stft_auto_btn.clicked.connect(self._reset_stft_axis_limits)
        self.thr_s1.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_s2.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_min_area_spin.valueChanged.connect(self._on_thr_params_changed)
        self.thr_keep_top_spin.valueChanged.connect(self._on_thr_params_changed)
        self.save_btn.clicked.connect(self._on_save_results)
        # --- connect 3D CWT/STFT widgets ---
        self.cwt_3d_db_cb.stateChanged.connect(self._on_cwt_3d_db_toggle)
        self.cwt_view_oblique.clicked.connect(lambda: self._set_cwt_3d_view('oblique'))
        self.cwt_view_iso.clicked.connect(lambda: self._set_cwt_3d_view('iso'))
        self.cwt_view_top.clicked.connect(lambda: self._set_cwt_3d_view('top'))

        self.stft_3d_db_cb.stateChanged.connect(self._on_stft_3d_db_toggle)
        self.stft_view_oblique.clicked.connect(lambda: self._set_stft_3d_view('oblique'))
        self.stft_view_iso.clicked.connect(lambda: self._set_stft_3d_view('iso'))
        self.stft_view_top.clicked.connect(lambda: self._set_stft_3d_view('top'))

        # ensure segmentation slider emits both label update and segment update on release
        try:
            self.seg_beat_slider.valueChanged.connect(self._on_seg_beat_changed)
            self.seg_beat_slider.sliderReleased.connect(self._on_seg_slider_released)
        except Exception:
            pass

    # ---------------- core actions ----------------
    def clear_all(self):
        # reset state & clear plots (similar to previous behavior)
        self.record_p_signal = None; self.fs = DEFAULT_FS; self.sig_names = []
        self.ecg_idx = None; self.pcg_idx = None; self.r_peaks = np.array([], dtype=int)
        self.current_segment = None; self.segment_bounds = (0,0)
        self.current_scalogram = None; self.current_freqs = None; self.current_times = None
        self.current_cwt_method = None; self.current_masks = {}; self.current_cogs = {}
        self.save_btn.setEnabled(False); self.swap_btn.setEnabled(False)
        self.pt_detect_btn.setEnabled(False); self.show_pt_pipeline_btn.setEnabled(False)
        self.seg_beat_slider.setEnabled(False)
        self.thr_s1.setEnabled(False); self.thr_s2.setEnabled(False)
        # clear and redraw canvases safely
        for fig_attr, canvas_attr in (('fig_pt_raw','canvas_pt_raw'),('fig_pt_proc','canvas_pt_proc'),('fig_pt_pcg','canvas_pt_pcg'),
                                     ('fig_seg_ecg','canvas_seg_ecg'),('fig_seg_pcg','canvas_seg_pcg'),
                                     ('fig_cwt','canvas_cwt'),('fig_cwt_3d','canvas_cwt_3d'),
                                     ('fig_stft_2d','canvas_stft_2d'),('fig_stft_3d','canvas_stft_3d'),('fig_thr','canvas_thr')):
            try:
                fig = getattr(self, fig_attr, None)
                if fig is not None:
                    fig.clf()
            except Exception:
                pass
        # re-create axes so canvases remain usable
        try:
            self.ax_pt_ecg_raw = self.fig_pt_raw.add_subplot(111); self._style_axis_dark(self.ax_pt_ecg_raw)
            self.ax_pt_ecg_proc = self.fig_pt_proc.add_subplot(111); self._style_axis_dark(self.ax_pt_ecg_proc)
            self.ax_pt_pcg = self.fig_pt_pcg.add_subplot(111); self._style_axis_dark(self.ax_pt_pcg)
            self.ax_seg_ecg = self.fig_seg_ecg.add_subplot(111); self._style_axis_dark(self.ax_seg_ecg)
            self.ax_seg_pcg = self.fig_seg_pcg.add_subplot(111); self._style_axis_dark(self.ax_seg_pcg)
            self.ax_cwt = self.fig_cwt.add_subplot(111); self._style_axis_dark(self.ax_cwt)
            try:
                self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111, projection='3d')
            except Exception:
                self.ax_cwt_3d = self.fig_cwt_3d.add_subplot(111)
            self._style_3d_axis(self.ax_cwt_3d)
            self.ax_stft_2d = self.fig_stft_2d.add_subplot(111); self._style_axis_dark(self.ax_stft_2d)
            try:
                self.ax_stft_3d = self.fig_stft_3d.add_subplot(111, projection='3d')
            except Exception:
                self.ax_stft_3d = self.fig_stft_3d.add_subplot(111)
            self._style_3d_axis(self.ax_stft_3d)
            self.ax_thr = self.fig_thr.add_subplot(111); self._style_axis_dark(self.ax_thr)
        except Exception:
            pass
        self.record_label.setText("No record loaded"); self.method_label.setText("CWT method: (not computed)")
        # redraw canvases
        for c in ('canvas_pt_raw','canvas_pt_proc','canvas_pt_pcg','canvas_seg_ecg','canvas_seg_pcg','canvas_cwt','canvas_cwt_3d','canvas_stft_2d','canvas_stft_3d','canvas_thr'):
            try:
                cc = getattr(self, c, None)
                if cc is not None:
                    cc.draw_idle()
            except Exception:
                pass
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
        self.pt_detect_btn.setEnabled(True); self.swap_btn.setEnabled(True)

    def _on_swap_channels(self):
        if self.record_p_signal is None:
            return
        self.ecg_idx, self.pcg_idx = self.pcg_idx, self.ecg_idx
        self.status.showMessage(f"Swapped: ECG={self.ecg_idx}, PCG={self.pcg_idx}")
        if self.r_peaks is not None and len(self.r_peaks) > 1:
            self._on_run_pt()

    # ---------------- Pan-Tompkins ----------------
    def _on_run_pt(self):
        if self.record_p_signal is None or self.ecg_idx is None:
            self.status.showMessage("Load a record and ensure ECG channel assigned.")
            return
        ecg = self.record_p_signal[:, self.ecg_idx].astype(float)
        try:
            r_peaks = detect_r_peaks_with_fallback(ecg, fs=self.fs, debug=False)
        except Exception as e:
            self.status.showMessage(f"PT detection error: {e}")
            print(traceback.format_exc()); return

        if r_peaks is None or len(r_peaks) < 2:
            self.status.showMessage("Too few R peaks detected.")
            return

        self.r_peaks = np.array(r_peaks, dtype=int)
        self.show_pt_pipeline_btn.setEnabled(True)
        # basic processed ECG visualization
        try:
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * self.fs
            b, a = butter(3, [5.0/nyq, 15.0/nyq], btype='band')
            ecg_filt = filtfilt(b, a, ecg)
            deriv = np.diff(ecg_filt, prepend=ecg_filt[0]); squared = deriv ** 2
            win = max(1, int(round(150.0 / 1000.0 * self.fs)))
            mwi = np.convolve(squared, np.ones(win)/win, mode='same')
        except Exception:
            ecg_filt = ecg.copy(); mwi = np.abs(ecg)

        pcg = self.record_p_signal[:, self.pcg_idx].astype(float)

        # PT plots
        try:
            self.ax_pt_ecg_raw.clear()
            t = np.arange(ecg.size) / float(self.fs)
            self.ax_pt_ecg_raw.plot(t, ecg, linewidth=0.6)
            self.ax_pt_ecg_raw.scatter(self.r_peaks / float(self.fs), ecg[self.r_peaks], c='r', s=10, label='R-peaks')
            self.ax_pt_ecg_raw.set_ylabel("ECG (raw)")
            self.ax_pt_ecg_raw.legend(fontsize='small')
            self._style_axis_dark(self.ax_pt_ecg_raw)
            self.canvas_pt_raw.draw_idle()
        except Exception:
            pass

        try:
            self.ax_pt_ecg_proc.clear()
            self.ax_pt_ecg_proc.plot(t, ecg_filt, linewidth=0.6, label='filtered')
            if np.max(mwi) != 0:
                self.ax_pt_ecg_proc.plot(t, mwi / (np.max(mwi) + 1e-12), label='MWI (norm)', alpha=0.8)
            self.ax_pt_ecg_proc.set_ylabel("Processed")
            self.ax_pt_ecg_proc.legend(fontsize='small')
            self._style_axis_dark(self.ax_pt_ecg_proc)
            self.canvas_pt_proc.draw_idle()
        except Exception:
            pass

        try:
            self.ax_pt_pcg.clear()
            t_pcg = np.arange(pcg.size) / float(self.fs)
            self.ax_pt_pcg.plot(t_pcg, pcg, linewidth=0.6)
            self.ax_pt_pcg.set_ylabel("PCG (raw)"); self.ax_pt_pcg.set_xlabel("Time [s]")
            self._style_axis_dark(self.ax_pt_pcg)
            self.canvas_pt_pcg.draw_idle()
        except Exception:
            pass

        # slider config
        n_beats = max(1, len(self.r_peaks) - 1)
        max_idx = max(0, n_beats - 1)
        self.seg_beat_slider.setMaximum(max_idx); self.seg_beat_slider.setEnabled(True)

        try:
            chosen_idx = choose_clean_beat(pcg, self.r_peaks, fs=self.fs)
            chosen_idx = int(chosen_idx)
        except Exception:
            chosen_idx = 0
        chosen_idx = max(0, min(chosen_idx, max_idx))
        self.seg_beat_slider.setValue(chosen_idx); self.seg_beat_label.setText(str(chosen_idx))

        self._update_segment_and_downstream(chosen_idx)
        self.status.showMessage(f"Pan-Tompkins: {len(self.r_peaks)} beats detected. (selected beat {chosen_idx})")

    def _on_show_pt_pipeline(self):
        if self.record_p_signal is None or self.ecg_idx is None:
            self.status.showMessage("Load record first.")
            return
        ecg = self.record_p_signal[:, self.ecg_idx].astype(float)
        before = set(plt.get_fignums())
        try:
            plot_pt_pipeline(ecg, fs=self.fs, r_peaks=self.r_peaks)
        except Exception as e:
            self.status.showMessage(f"plot_pt_pipeline error: {e}")
            print(traceback.format_exc()); return
        after = set(plt.get_fignums()) - before
        for num in after:
            try:
                fig = plt.figure(num)
                fig.patch.set_facecolor(FIG_BG)
                for ax in fig.axes:
                    try:
                        ax.set_facecolor(AX_BG)
                        for spine in ax.spines.values():
                            spine.set_color('#6e9fbf')
                        ax.tick_params(colors='#cfe8ff', which='both')
                        ax.xaxis.label.set_color('#dbeeff'); ax.yaxis.label.set_color('#dbeeff')
                        ax.title.set_color('#eaf6ff')
                        try: fig.tight_layout()
                        except Exception: pass
                    except Exception:
                        pass
                try: fig.canvas.draw_idle()
                except Exception:
                    try: fig.canvas.draw()
                    except Exception: pass
            except Exception:
                pass
        self.status.showMessage("Shown PT pipeline (styled to dark theme).")

    # ---------- segmentation & downstream ----------
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

        self.current_segment = seg; self.segment_bounds = (start, end)

        # segmentation plots
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
                self.ax_seg_ecg.set_title("ECG around selected R"); self.ax_seg_ecg.set_ylabel("ECG")
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
            self.ax_seg_pcg.set_title("PCG segment"); self.ax_seg_pcg.set_ylabel("PCG"); self.ax_seg_pcg.set_xlabel("Time [s]")
            self._style_axis_dark(self.ax_seg_pcg)
            try: self.fig_seg_pcg.tight_layout()
            except Exception: pass
            self.canvas_seg_pcg.draw_idle()
        except Exception:
            pass

        # read TF parameters
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            fmin, fmax, nfreqs = DEFAULT_CWT_FMIN, DEFAULT_CWT_FMAX, DEFAULT_CWT_NFREQS

        backend_sel = str(self.cwt_backend_combo.currentText()).strip().lower()
        tf_method = str(self.tf_method_combo.currentText()).strip().upper()

        # compute accordingly
        if tf_method == 'CWT':
            compute_kwargs = {}
            if backend_sel == 'pascal':
                compute_kwargs['a0'] = float(self.cwt_a0_spin.value())
                compute_kwargs['a_step'] = float(self.cwt_astep_spin.value())
                compute_kwargs['col_count'] = int(self.cwt_colcount_spin.value())
                if self.cwt_use_freqs_target.isChecked():
                    compute_kwargs['freqs_target'] = np.linspace(fmin, fmax, nfreqs)
            try:
                scalogram, freqs, times_rel, method = compute_cwt(seg, fs=self.fs, fmin=fmin, fmax=fmax, n_freqs=nfreqs, backend=backend_sel, **compute_kwargs)
            except Exception as e:
                self.status.showMessage(f"CWT error: {e}"); print(traceback.format_exc()); return
            start_sample, _ = self.segment_bounds
            times_abs = times_rel + start_sample / float(self.fs) if (times_rel is not None and len(times_rel) > 0) else np.array([])
            try:
                scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(scalogram, freqs, times_abs)
            except Exception:
                self.status.showMessage("CWT returned invalid scalogram shape"); return
            self.current_scalogram = scal2; self.current_freqs = freqs2; self.current_times = times2; self.current_cwt_method = method
            self.method_label.setText(f"CWT method: {method} (backend: {backend_sel})")
            self._update_masks_and_cogs()
            self._plot_cwt_tab(); self._plot_cwt_3d(); self._plot_threshold_tab()
            self.thr_s1.setEnabled(True); self.thr_s2.setEnabled(True); self.save_btn.setEnabled(True)
            self.status.showMessage(f"Segment updated. CWT method: {method}")
        else:  # STFT
            if compute_stft is None:
                self.status.showMessage("STFT backend not available (STFT_PCG.py missing)."); return
            try:
                res = compute_stft(seg, fs=self.fs, nperseg=int(self.stft_nperseg.value()), noverlap=int(self.stft_noverlap.value()))
                stft_Sxx, stft_freqs, stft_times, stft_method = self._unpack_stft_result(res)
            except Exception as e:
                self.status.showMessage(f"STFT error: {e}"); print(traceback.format_exc()); return
            # Use STFT outputs; set display times to match CWT's time range if CWT exists (per request)
            self.current_scalogram = np.abs(stft_Sxx); self.current_freqs = stft_freqs
            start_sample, _ = self.segment_bounds
            # times returned by STFT are relative to segment start (likely), but we want consistent absolute times
            if stft_times is not None and len(stft_times) > 0:
                self.current_times = stft_times + start_sample / float(self.fs)
            else:
                self.current_times = np.linspace(start_sample/float(self.fs), (start_sample + len(seg)-1)/float(self.fs), self.current_scalogram.shape[1] if self.current_scalogram is not None else 1)
            self.current_cwt_method = stft_method or 'STFT'
            self.method_label.setText(f"TF method: {self.current_cwt_method}")
            # update axis controls default to match CWT 2D times if we have a CWT computed earlier
            # Requirement: "set duration/time default same with chart in CWT 2D"
            if getattr(self, 'current_times', None) is not None:
                t0, t1 = float(self.current_times[0]), float(self.current_times[-1])
                try:
                    self.stft_xmin.blockSignals(True); self.stft_xmax.blockSignals(True)
                    self.stft_xmin.setRange(t0 - 10.0, t1 + 10.0); self.stft_xmax.setRange(t0 - 10.0, t1 + 10.0)
                    self.stft_xmin.setValue(t0); self.stft_xmax.setValue(t1)
                finally:
                    self.stft_xmin.blockSignals(False); self.stft_xmax.blockSignals(False)
            if getattr(self, 'current_freqs', None) is not None:
                f0, f1 = float(np.min(self.current_freqs)), float(np.max(self.current_freqs))
                try:
                    self.stft_ymin.blockSignals(True); self.stft_ymax.blockSignals(True)
                    self.stft_ymin.setRange(0.0, f1 + 100.0); self.stft_ymax.setRange(0.0, f1 + 100.0)
                    self.stft_ymin.setValue(f0); self.stft_ymax.setValue(f1)
                finally:
                    self.stft_ymin.blockSignals(False); self.stft_ymax.blockSignals(False)

            self._update_masks_and_cogs()
            # plot STFT in its own tab
            self._plot_stft_tab(); self._plot_stft_3d(); self._plot_threshold_tab()
            self.thr_s1.setEnabled(True); self.thr_s2.setEnabled(True); self.save_btn.setEnabled(True)
            self.status.showMessage("Segment updated. STFT computed.")

    # ---------------- helper: scalogram shape fixes ----------------
    def _ensure_scalogram_orientation_and_axes(self, scal, freqs, times):
        scal2 = np.asarray(scal)
        if scal2.ndim != 2:
            raise ValueError("scalogram must be 2D")
        nrows, ncols = scal2.shape
        # freqs
        if freqs is None or len(freqs) == 0:
            freqs = np.linspace(0.0, 1.0, nrows)
        freqs = np.asarray(freqs)
        if freqs.size != nrows:
            f0 = float(freqs[0]) if freqs.size>0 else 0.0
            fend = float(freqs[-1]) if freqs.size>1 else f0 + max(1.0, nrows/10.0)
            freqs = np.linspace(f0, fend, nrows)
        if freqs[0] > freqs[-1]:
            scal2 = scal2[::-1, :]; freqs = freqs[::-1]
        # times
        if times is None or len(times) == 0:
            times = np.linspace(0.0, float(ncols-1)/float(self.fs), ncols)
        if len(times) != ncols:
            t0 = float(times[0]) if len(times)>0 else 0.0
            tend = float(times[-1]) if len(times)>1 else t0 + float(ncols-1)/float(self.fs)
            times = np.linspace(t0, tend, ncols)
        return scal2, freqs, times

    # ---------------- CWT plotting (2D) ----------------
    def _plot_cwt_tab(self):
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            try:
                self.ax_cwt.clear(); self.canvas_cwt.draw_idle()
            except Exception:
                pass
            return
        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(self.current_scalogram, self.current_freqs, self.current_times)
        except Exception:
            scal = self.current_scalogram; freqs = self.current_freqs; times = self.current_times
        self.current_scalogram = scal; self.current_freqs = freqs; self.current_times = times
        try:
            for ax in list(self.fig_cwt.axes):
                if ax is not self.ax_cwt:
                    try: self.fig_cwt.delaxes(ax)
                    except Exception: pass
        except Exception:
            pass
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        self.ax_cwt.clear()
        self.cwt_im = self.ax_cwt.imshow(scal, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
        try:
            if getattr(self, 'cwt_cb', None) is not None:
                try: self.cwt_cb.remove()
                except Exception: pass
            divider = make_axes_locatable(self.ax_cwt)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            self.cwt_cb = self.fig_cwt.colorbar(self.cwt_im, cax=cax)
            self.cwt_cax = cax
        except Exception:
            self.cwt_cb = None; self.cwt_cax = None
        try:
            self.ax_cwt.set_xlim(times[0], times[-1]); self.ax_cwt.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass
        self.ax_cwt.set_xlabel("Time [s]"); self.ax_cwt.set_ylabel("Frequency [Hz]")
        self.ax_cwt.set_title(f"Scalogram ({self.current_cwt_method})")
        self._style_axis_dark(self.ax_cwt)
        try: self.fig_cwt.tight_layout()
        except Exception: pass
        self.canvas_cwt.draw_idle()

    # ---------------- 3D plotting for CWT ----------------
    def _plot_cwt_3d(self, max_surface_points=120000, cmap='viridis', elev=35, azim=-125, use_db=False):
        """
        Plot a 3D surface of the current scalogram (time x freq x magnitude).
        - use_db True by default for dynamic range
        - view_init chosen to show time axis forward, freq axis to right, z up.
        """
        try:
            if self.current_scalogram is None:
                return
            scal = np.asarray(self.current_scalogram)
            freqs = np.asarray(self.current_freqs) if self.current_freqs is not None else None
            times = np.asarray(self.current_times) if self.current_times is not None else None
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(scal, freqs, times)

            T, F = np.meshgrid(times, freqs)
            # inside _plot_cwt_3d:
            use_db = getattr(self, 'cwt_3d_use_db', True)
            Zp = self.current_scalogram.copy()
            if use_db:
                with np.errstate(divide='ignore'):
                    Zp = 20.0 * np.log10(np.maximum(Zp, 1e-12))
            # then plot Z on the surface
            
            #Z = np.abs(scal)
            #Zp = 20.0 * np.log10(Z + 1e-12) if use_db else Z

            # normalize to 2D colormap dynamic range for similar appearance
            norm = Normalize(vmin=np.nanpercentile(Zp, 5), vmax=np.nanpercentile(Zp, 99))

            # downsample adaptively to keep interactive
            n_total = Zp.size
            if n_total > max_surface_points:
                factor = int(np.ceil(np.sqrt(float(n_total)/float(max_surface_points))))
                f_idx = np.arange(0, Zp.shape[0], factor); t_idx = np.arange(0, Zp.shape[1], factor)
                Ts = T[np.ix_(f_idx, t_idx)]; Fs = F[np.ix_(f_idx, t_idx)]; Zs = Zp[np.ix_(f_idx, t_idx)]
            else:
                Ts, Fs, Zs = T, F, Zp

            # clear and plot
            try: self.ax_cwt_3d.cla()
            except Exception: pass
            try:
                surf = self.ax_cwt_3d.plot_surface(Ts, Fs, Zs, rstride=1, cstride=1,
                                                   facecolors=cm.get_cmap(cmap)(norm(Zs)),
                                                   linewidth=0, antialiased=True, shade=False)
                # colorbar onto a small axis (preserve main axis)
                try:
                    if hasattr(self, 'cwt_3d_cb_ax') and (self.cwt_3d_cb_ax in self.fig_cwt_3d.axes):
                        try: self.fig_cwt_3d.delaxes(self.cwt_3d_cb_ax)
                        except Exception: pass
                    cax = self.fig_cwt_3d.add_axes([0.92, 0.15, 0.02, 0.7])
                    cb = self.fig_cwt_3d.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                    cb.ax.yaxis.set_tick_params(labelcolor='white')
                    self.cwt_3d_cb_ax = cax
                except Exception:
                    pass
            except Exception:
                try:
                    self.ax_cwt_3d.plot_wireframe(Ts, Fs, Zs)
                except Exception:
                    pass

            # labels, orientation and style
            try:
                self.ax_cwt_3d.set_xlabel("Time [s]"); self.ax_cwt_3d.set_ylabel("Frequency [Hz]"); self.ax_cwt_3d.set_zlabel("Magnitude (dB)" if use_db else "Magnitude")
                self.ax_cwt_3d.set_title(f"3D CWT ({self.current_cwt_method or 'CWT'})")
                try: self.ax_cwt_3d.view_init(elev=elev, azim=azim)
                except Exception: pass
            except Exception:
                pass
            try: self._style_3d_axis(self.ax_cwt_3d)
            except Exception: pass
            try: self.canvas_cwt_3d.draw_idle()
            except Exception:
                try: self.fig_cwt_3d.canvas.draw()
                except Exception: pass
        except Exception as e:
            print("Warning: _plot_cwt_3d failed:", e)
            try:
                import traceback as _tb
                _tb.print_exc()
            except Exception:
                pass
            return

    # ---------- STFT plotting (2D + 3D) ----------
    def _plot_stft_tab(self):
        # Accepts current_scalogram, current_freqs, current_times
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

        # use the current_times/current_freqs (ensured by callers)
        try:
            for ax in list(self.fig_stft_2d.axes):
                if ax is not self.ax_stft_2d:
                    try: self.fig_stft_2d.delaxes(ax)
                    except Exception: pass
        except Exception:
            pass

        # determine extent: prefer CWT times if set by user earlier (per request current_times is set to segment/time)
        extent = [times[0], times[-1], freqs[0], freqs[-1]]

        self.ax_stft_2d.clear()
        # show in dB for readable dynamic range
        #use_db = getattr(self, 'stft_3d_use_db', True)
        #Zp = self.current_scalogram.copy()
        Zp = np.abs(S)
        '''if use_db:
            with np.errstate(divide='ignore'):
                Zp = 20.0 * np.log10(np.maximum(Zp, 1e-12))'''
        #Z = np.abs(S)
        #Zp = 20.0 * np.log10(Z + 1e-12)
        im = self.ax_stft_2d.imshow(Zp, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
        try:
            # colorbar
            if getattr(self, 'stft_cb', None) is not None:
                try: self.stft_cb.remove()
                except Exception: pass
            divider = make_axes_locatable(self.ax_stft_2d)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            self.stft_cb = self.fig_stft_2d.colorbar(im, cax=cax)
            self.stft_cax = cax
        except Exception:
            self.stft_cb = None; self.stft_cax = None

        # Apply any axis limits set by controls (if user applied)
        try:
            if self.stft_xmin.value() < self.stft_xmax.value():
                self.ax_stft_2d.set_xlim(float(self.stft_xmin.value()), float(self.stft_xmax.value()))
            else:
                self.ax_stft_2d.set_xlim(times[0], times[-1])
            if self.stft_ymin.value() < self.stft_ymax.value():
                self.ax_stft_2d.set_ylim(float(self.stft_ymin.value()), float(self.stft_ymax.value()))
            else:
                self.ax_stft_2d.set_ylim(freqs[0], freqs[-1])
        except Exception:
            try: self.ax_stft_2d.set_xlim(times[0], times[-1]); self.ax_stft_2d.set_ylim(freqs[0], freqs[-1])
            except Exception: pass

        self.ax_stft_2d.set_xlabel("Time [s]"); self.ax_stft_2d.set_ylabel("Frequency [Hz]")
        self.ax_stft_2d.set_title("STFT Spectrogram (for dB use must be changed within the GUI.py or hardcoded)")
        self._style_axis_dark(self.ax_stft_2d)
        try:
            self.fig_stft_2d.tight_layout()
        except Exception:
            pass
        self.canvas_stft_2d.draw_idle()

    def _plot_stft_3d(self, max_surface_points=120000, cmap='viridis', elev=35, azim=-125, use_db=False):
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            try:
                self.ax_stft_3d.clear(); self.canvas_stft_3d.draw_idle()
            except Exception:
                pass
            return
        S = np.asarray(self.current_scalogram)
        freqs = np.asarray(self.current_freqs)
        times = np.asarray(self.current_times)
        try:
            S, freqs, times = self._ensure_scalogram_orientation_and_axes(S, freqs, times)
        except Exception:
            pass

        T, F = np.meshgrid(times, freqs)
        # inside _plot_cwt_3d:
        use_db = getattr(self, 'stft_3d_use_db', True)
        #Zp = self.current_scalogram.copy()
        Zp= np.abs(S)
        if use_db:
            with np.errstate(divide='ignore'):
                Zp = 20.0 * np.log10(np.maximum(Zp, 1e-12))
        # then plot Z on the surface

        #Z = np.abs(S)
        #Zp = 20.0 * np.log10(Z + 1e-12) if use_db else Z
        # Normalize similar to 2D
        norm = Normalize(vmin=np.nanpercentile(Zp, 5), vmax=np.nanpercentile(Zp, 99))

        # downsample if too many points
        n_total = Zp.size
        if n_total > max_surface_points:
            factor = int(np.ceil(np.sqrt(float(n_total)/float(max_surface_points))))
            f_idx = np.arange(0, Zp.shape[0], factor); t_idx = np.arange(0, Zp.shape[1], factor)
            Ts = T[np.ix_(f_idx, t_idx)]; Fs = F[np.ix_(f_idx, t_idx)]; Zs = Zp[np.ix_(f_idx, t_idx)]
        else:
            Ts, Fs, Zs = T, F, Zp

        try: self.ax_stft_3d.cla()
        except Exception: pass
        try:
            surf = self.ax_stft_3d.plot_surface(Ts, Fs, Zs,
                                                facecolors=cm.get_cmap(cmap)(norm(Zs)),
                                                rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
            try:
                if hasattr(self, 'stft_3d_cb_ax') and (self.stft_3d_cb_ax in self.fig_stft_3d.axes):
                    try: self.fig_stft_3d.delaxes(self.stft_3d_cb_ax)
                    except Exception: pass
                cax = self.fig_stft_3d.add_axes([0.92, 0.15, 0.02, 0.7])
                cb = self.fig_stft_3d.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                cb.ax.yaxis.set_tick_params(labelcolor='white')
                self.stft_3d_cb_ax = cax
            except Exception:
                pass
        except Exception:
            try: self.ax_stft_3d.plot_wireframe(Ts, Fs, Zs)
            except Exception: pass

        # Apply axis limits if user set them
        try:
            if self.stft_xmin.value() < self.stft_xmax.value():
                self.ax_stft_3d.set_xlim(float(self.stft_xmin.value()), float(self.stft_xmax.value()))
            else:
                self.ax_stft_3d.set_xlim(times[0], times[-1])
            if self.stft_ymin.value() < self.stft_ymax.value():
                self.ax_stft_3d.set_ylim(float(self.stft_ymin.value()), float(self.stft_ymax.value()))
            else:
                self.ax_stft_3d.set_ylim(freqs[0], freqs[-1])
        except Exception:
            try: self.ax_stft_3d.set_xlim(times[0], times[-1]); self.ax_stft_3d.set_ylim(freqs[0], freqs[-1])
            except Exception: pass

        try:
            self.ax_stft_3d.set_xlabel("Time [s]"); self.ax_stft_3d.set_ylabel("Frequency [Hz]"); self.ax_stft_3d.set_zlabel("Magnitude (dB)" if use_db else "Magnitude")
            self.ax_stft_3d.set_title("3D STFT")
            try: self.ax_stft_3d.view_init(elev=elev, azim=azim)
            except Exception: pass
        except Exception:
            pass
        try: self._style_3d_axis(self.ax_stft_3d)
        except Exception: pass
        try: self.canvas_stft_3d.draw_idle()
        except Exception:
            try: self.fig_stft_3d.canvas.draw()
            except Exception: pass

    # ---------------- STFT result unpack helper ----------------
    def _unpack_stft_result(self, res):
        """Accept either (Sxx, freqs, times) or (Sxx, freqs, times, method). Return 4-tuple."""
        if res is None:
            raise ValueError("STFT returned None")
        if isinstance(res, (list, tuple)):
            if len(res) == 3:
                Sxx, freqs, times = res; method = "STFT"
            elif len(res) == 4:
                Sxx, freqs, times, method = res
            else:
                raise ValueError(f"Unexpected return shape from compute_stft (len={len(res)})")
            return Sxx, np.asarray(freqs, dtype=float), np.asarray(times, dtype=float), str(method)
        raise ValueError("Unexpected STFT result type")

    # ---------------- Threshold & CoG ----------------
    def _get_thr_params(self):
        ma = int(self.thr_min_area_spin.value()); min_area = None if ma == 0 else ma
        keep_top = int(self.thr_keep_top_spin.value()); return min_area, keep_top

    def _apply_time_window_filter(self, mask, times, time_window):
        if mask is None: return mask
        mask = np.asarray(mask, dtype=bool)
        if times is None or time_window is None: return mask
        t0, t1 = float(time_window[0]), float(time_window[1])
        if sp_ndimage is not None:
            labeled, ncomp = sp_ndimage.label(mask)
            if ncomp == 0: return np.zeros_like(mask, dtype=bool)
            keep_mask = np.zeros_like(mask, dtype=bool)
            for comp in range(1, ncomp+1):
                comp_mask = (labeled == comp)
                if not comp_mask.any(): continue
                try:
                    centroid = sp_ndimage.center_of_mass(comp_mask)
                    _, colc = centroid
                except Exception:
                    ys, xs = np.nonzero(comp_mask); colc = float(xs.mean()) if xs.size else 0.0
                col_idx = int(round(colc)); col_idx = max(0, min(col_idx, len(times)-1))
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
        s1 = self.thr_s1.value() / 100.0; s2 = self.thr_s2.value() / 100.0
        min_area, keep_top = self._get_thr_params()
        try:
            mask1 = threshold_mask(self.current_scalogram, s1, min_area=min_area, keep_top=keep_top)
            mask2 = threshold_mask(self.current_scalogram, s2, min_area=min_area, keep_top=keep_top)
            if mask1 is not None and mask1.shape != self.current_scalogram.shape:
                if mask1.T.shape == self.current_scalogram.shape: mask1 = mask1.T
            if mask2 is not None and mask2.shape != self.current_scalogram.shape:
                if mask2.T.shape == self.current_scalogram.shape: mask2 = mask2.T
            start_sample, end_sample = self.segment_bounds
            seg_dur_s = float(end_sample - start_sample) / float(self.fs) if (end_sample > start_sample) else (len(self.current_segment)/float(self.fs) if self.current_segment is not None else 0.0)
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
                    try: self.fig_thr.delaxes(ax)
                    except Exception: pass
        except Exception:
            pass
        if getattr(self, 'thr_cb', None) is not None:
            try: self.thr_cb.remove()
            except Exception: pass
            self.thr_cb = None
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            try: self.canvas_thr.draw_idle()
            except Exception: pass
            return
        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(self.current_scalogram, self.current_freqs, self.current_times)
        except Exception:
            scal = self.current_scalogram; freqs = self.current_freqs; times = self.current_times
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        self.ax_thr.clear()
        self.thr_im = self.ax_thr.imshow(scal, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
        try:
            divider = make_axes_locatable(self.ax_thr)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            self.thr_cb = self.fig_thr.colorbar(self.thr_im, cax=cax)
            self.thr_cax = cax
        except Exception:
            self.thr_cb = None; self.thr_cax = None
        mask1 = self.current_masks.get('S1', None); mask2 = self.current_masks.get('S2', None)
        def safe_contour(ax, times_arr, freqs_arr, mask_arr, **kwargs):
            if mask_arr is None: return None
            mask_arr = np.asarray(mask_arr)
            if mask_arr.shape != (len(freqs_arr), len(times_arr)):
                if mask_arr.T.shape == (len(freqs_arr), len(times_arr)):
                    mask_arr = mask_arr.T
                else: return None
            try:
                c = ax.contour(times_arr, freqs_arr, mask_arr.astype(int), levels=[0.5], **kwargs)
                return c
            except Exception:
                return None
        if mask1 is not None and np.any(mask1):
            safe_contour(self.ax_thr, times, freqs, mask1, colors='white', linewidths=1)
        if mask2 is not None and np.any(mask2):
            safe_contour(self.ax_thr, times, freqs, mask2, colors='cyan', linewidths=1)
        cog1 = self.current_cogs.get('S1', None); cog2 = self.current_cogs.get('S2', None)
        if cog1 is not None:
            try: self.ax_thr.scatter([cog1[0]], [cog1[1]], marker='x', s=80, c='white', zorder=5, label='CoG S1')
            except Exception: pass
        if cog2 is not None:
            try: self.ax_thr.scatter([cog2[0]], [cog2[1]], marker='o', s=80, edgecolors='cyan', facecolors='none', zorder=5, label='CoG S2')
            except Exception: pass
        try:
            legends = []
            if cog1 is not None: legends.append(Line2D([0],[0], marker='x', color='w', label='CoG S1', linestyle=''))
            if cog2 is not None: legends.append(Line2D([0],[0], marker='o', markerfacecolor='none', markeredgecolor='c', label='CoG S2', linestyle=''))
            if legends: self.ax_thr.legend(handles=legends, loc='upper right', fontsize='small')
        except Exception:
            pass
        try:
            self.ax_thr.set_xlim(times[0], times[-1]); self.ax_thr.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass
        self.ax_thr.set_xlabel("Time [s]"); self.ax_thr.set_ylabel("Frequency [Hz]")
        self._style_axis_dark(self.ax_thr)
        try: self.fig_thr.tight_layout()
        except Exception: pass
        self.canvas_thr.draw_idle()

    # ---------------- UI events ----------------
    def _on_cwt_params_changed(self):
        if self.current_segment is None:
            return
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            self.status.showMessage("Invalid CWT parameters"); return
        self._compute_cwt_for_current_segment(fmin, fmax, nfreqs)

    def _compute_cwt_for_current_segment(self, fmin, fmax, nfreqs):
        if self.current_segment is None: return
        backend_sel = str(self.cwt_backend_combo.currentText()).strip().lower()
        tf_method = str(self.tf_method_combo.currentText()).strip().upper()
        compute_kwargs = {}
        if backend_sel == 'pascal':
            compute_kwargs['a0'] = float(self.cwt_a0_spin.value()); compute_kwargs['a_step'] = float(self.cwt_astep_spin.value()); compute_kwargs['col_count'] = int(self.cwt_colcount_spin.value())
            if self.cwt_use_freqs_target.isChecked():
                compute_kwargs['freqs_target'] = np.linspace(fmin, fmax, nfreqs)
        if tf_method == 'CWT':
            try:
                scalogram, freqs, times_rel, method = compute_cwt(self.current_segment, fs=self.fs, fmin=fmin, fmax=fmax, n_freqs=nfreqs, backend=backend_sel, **compute_kwargs)
            except Exception as e:
                self.status.showMessage(f"CWT error: {e}"); print(traceback.format_exc()); return
            start_sample, _ = self.segment_bounds
            times_abs = times_rel + start_sample / float(self.fs) if (times_rel is not None and len(times_rel)>0) else np.array([])
            try:
                scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(scalogram, freqs, times_abs)
            except Exception:
                self.status.showMessage("CWT returned invalid scalogram shape"); return
            self.current_scalogram = scal2; self.current_freqs = freqs2; self.current_times = times2; self.current_cwt_method = method
            self.method_label.setText(f"CWT method: {method} (backend: {backend_sel})")
            self._update_masks_and_cogs(); self._plot_cwt_tab(); self._plot_cwt_3d(); self._plot_threshold_tab()
            self.status.showMessage(f"CWT recomputed ({method}) using backend={backend_sel}")
        else:
            if compute_stft is None:
                self.status.showMessage("STFT backend not available"); return
            try:
                res = compute_stft(self.current_segment, fs=self.fs, nperseg=int(self.stft_nperseg.value()), noverlap=int(self.stft_noverlap.value()))
                stft_Sxx, stft_freqs, stft_times, stft_method = self._unpack_stft_result(res)
            except Exception as e:
                self.status.showMessage(f"STFT error: {e}"); print(traceback.format_exc()); return
            self.current_scalogram = np.abs(stft_Sxx); self.current_freqs = stft_freqs
            start_sample, _ = self.segment_bounds
            self.current_times = stft_times + start_sample / float(self.fs) if (stft_times is not None and len(stft_times)>0) else None
            self.current_cwt_method = stft_method or 'STFT'; self.method_label.setText("TF method: STFT")
            # set axis controls default
            if getattr(self, 'current_times', None) is not None:
                t0, t1 = float(self.current_times[0]), float(self.current_times[-1])
                try:
                    self.stft_xmin.setRange(t0 - 10.0, t1 + 10.0); self.stft_xmax.setRange(t0 - 10.0, t1 + 10.0)
                    self.stft_xmin.setValue(t0); self.stft_xmax.setValue(t1)
                except Exception:
                    pass
            if getattr(self, 'current_freqs', None) is not None:
                f0, f1 = float(np.min(self.current_freqs)), float(np.max(self.current_freqs))
                try:
                    self.stft_ymin.setRange(0.0, f1 + 100.0); self.stft_ymax.setRange(0.0, f1 + 100.0)
                    self.stft_ymin.setValue(f0); self.stft_ymax.setValue(f1)
                except Exception:
                    pass
            self._update_masks_and_cogs(); self._plot_stft_tab(); self._plot_stft_3d(); self._plot_threshold_tab()
            self.status.showMessage("STFT recomputed")

    def _on_stft_params_changed(self, _v=None):
        if self.current_segment is None: return
        if str(self.tf_method_combo.currentText()).strip().upper() == 'STFT':
            try:
                self._compute_cwt_for_current_segment(float(self.cwt_fmin.text()), float(self.cwt_fmax.text()), int(self.cwt_nfreqs.text()))
            except Exception:
                # fallback: simply recompute STFT via update
                self._update_segment_and_downstream(int(self.seg_beat_slider.value()))

    def _on_thr_slider_changed(self, _val):
        self.thr_s1_label.setText(f"{self.thr_s1.value()/100.0:.2f}")
        self.thr_s2_label.setText(f"{self.thr_s2.value()/100.0:.2f}")
        self._update_masks_and_cogs(); self._plot_threshold_tab()

    def _on_thr_params_changed(self, _v=None):
        if self.current_scalogram is None: return
        self._update_masks_and_cogs(); self._plot_threshold_tab()

    # ---------------- STFT axis controls ----------------
    def _apply_stft_axis_limits(self):
        """Apply the user-specified STFT axis limits to both 2D and 3D STFT axes."""
        try:
            # apply to 2D
            if getattr(self, 'ax_stft_2d', None) is not None:
                if self.stft_xmin.value() < self.stft_xmax.value():
                    self.ax_stft_2d.set_xlim(float(self.stft_xmin.value()), float(self.stft_xmax.value()))
                if self.stft_ymin.value() < self.stft_ymax.value():
                    self.ax_stft_2d.set_ylim(float(self.stft_ymin.value()), float(self.stft_ymax.value()))
                self.canvas_stft_2d.draw_idle()
            # apply to 3D
            if getattr(self, 'ax_stft_3d', None) is not None:
                if self.stft_xmin.value() < self.stft_xmax.value():
                    try: self.ax_stft_3d.set_xlim(float(self.stft_xmin.value()), float(self.stft_xmax.value()))
                    except Exception: pass
                if self.stft_ymin.value() < self.stft_ymax.value():
                    try: self.ax_stft_3d.set_ylim(float(self.stft_ymin.value()), float(self.stft_ymax.value()))
                    except Exception: pass
                try: self.canvas_stft_3d.draw_idle()
                except Exception: pass
        except Exception:
            pass

    def _reset_stft_axis_limits(self):
        """Reset STFT axis limits to automatic (extent from current_times/current_freqs)"""
        try:
            if getattr(self, 'current_times', None) is not None:
                t0, t1 = float(self.current_times[0]), float(self.current_times[-1])
                self.stft_xmin.setValue(t0); self.stft_xmax.setValue(t1)
            if getattr(self, 'current_freqs', None) is not None:
                f0, f1 = float(np.min(self.current_freqs)), float(np.max(self.current_freqs))
                self.stft_ymin.setValue(f0); self.stft_ymax.setValue(f1)
            # re-plot with auto extents
            self._plot_stft_tab(); self._plot_stft_3d()
        except Exception:
            pass

    # ---------------- save ----------------
    def _on_save_results(self):
        if self.current_scalogram is None or self.current_segment is None:
            self.status.showMessage("No result to save"); return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder to save results", os.getcwd())
        if not folder: return
        base = (self.record_label.text().split()[1] if "Loaded:" in self.record_label.text() else "record")
        beat_idx = int(self.seg_beat_slider.value()) if self.seg_beat_slider.isEnabled() else 0
        s1 = self.thr_s1.value() / 100.0; s2 = self.thr_s2.value() / 100.0
        min_area, keep_top = self._get_thr_params()
        filename_base = f"{base}_beat{beat_idx}_s1_{int(s1*100)}_s2_{int(s2*100)}"
        png_path = os.path.join(folder, filename_base + ".png"); json_path = os.path.join(folder, filename_base + ".json")
        try:
            self.fig_thr.savefig(png_path, dpi=300, bbox_inches='tight')
            metadata = {
                'record': base, 'fs': int(self.fs), 'pcg_channel': int(self.pcg_idx),
                'ecg_channel': int(self.ecg_idx) if self.ecg_idx is not None else None,
                'beat_index': beat_idx,
                'segment_samples': {'start': int(self.segment_bounds[0]), 'end': int(self.segment_bounds[1])},
                'cwt_method': str(self.current_cwt_method),
                'cwt_params': {
                    'fmin': float(self.cwt_fmin.text()), 'fmax': float(self.cwt_fmax.text()), 'n_freqs': int(self.cwt_nfreqs.text()),
                    'backend': str(self.cwt_backend_combo.currentText()), 'use_explicit_freqs': bool(self.cwt_use_freqs_target.isChecked()),
                    'col_count': int(self.cwt_colcount_spin.value()), 'a0': float(self.cwt_a0_spin.value()), 'a_step': float(self.cwt_astep_spin.value())
                },
                'thresholds': {'S1': s1, 'S2': s2, 'min_area': min_area, 'keep_top': keep_top},
                'CoG': {
                    'S1': None if self.current_cogs.get('S1') is None else {'t_s': float(self.current_cogs['S1'][0]), 'f_hz': float(self.current_cogs['S1'][1])},
                    'S2': None if self.current_cogs.get('S2') is None else {'t_s': float(self.current_cogs['S2'][0]), 'f_hz': float(self.current_cogs['S2'][1])},
                }
            }
            with open(json_path, 'w') as f: json.dump(metadata, f, indent=2)
            self.status.showMessage(f"Saved PNG: {png_path} and JSON: {json_path}")
        except Exception as e:
            self.status.showMessage(f"Save failed: {e}"); print(traceback.format_exc())
    
    # ---------------- 3D controls handlers ----------------
    def _on_cwt_3d_db_toggle(self, state):
        """Replot CWT 3D using dB scale if checked."""
        try:
            self.cwt_3d_use_db = (state == QtCore.Qt.CheckState.Checked or bool(state))
        except Exception:
            self.cwt_3d_use_db = bool(state)
        # If plotting function exists, call it
        try:
            if hasattr(self, '_plot_cwt_3d'):
                self._plot_cwt_3d()
        except Exception:
            print("Warning: _plot_cwt_3d failed in _on_cwt_3d_db_toggle", traceback.format_exc())

    def _on_stft_3d_db_toggle(self, state):
        """Replot STFT 3D using dB scale if checked."""
        try:
            self.stft_3d_use_db = (state == QtCore.Qt.CheckState.Checked or bool(state))
        except Exception:
            self.stft_3d_use_db = bool(state)
        try:
            if hasattr(self, '_plot_stft_3d'):
                self._plot_stft_3d()
        except Exception:
            print("Warning: _plot_stft_3d failed in _on_stft_3d_db_toggle", traceback.format_exc())

    def _set_cwt_3d_view(self, preset):
        """Set preset 3D view for CWT 3D Axes."""
        presets = {'oblique': (35, -125), 'iso': (45, -135), 'top': (90, -90)}
        elev, azim = presets.get(preset, (35, -125))
        try:
            ax = getattr(self, 'ax_cwt_3d', None)
            if ax is not None and hasattr(ax, 'view_init'):
                ax.view_init(elev=elev, azim=azim)
                try: self.canvas_cwt_3d.draw_idle()
                except Exception: pass
        except Exception:
            print("Warning: _set_cwt_3d_view", traceback.format_exc())

    def _set_stft_3d_view(self, preset):
        """Set preset 3D view for STFT 3D Axes."""
        presets = {'oblique': (35, -125), 'iso': (45, -135), 'top': (90, -90)}
        elev, azim = presets.get(preset, (35, -125))
        try:
            ax = getattr(self, 'ax_stft_3d', None)
            if ax is not None and hasattr(ax, 'view_init'):
                ax.view_init(elev=elev, azim=azim)
                try: self.canvas_stft_3d.draw_idle()
                except Exception: pass
        except Exception:
            print("Warning: _set_stft_3d_view", traceback.format_exc())


# ---------------- run helper ----------------
def run_app():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = PCGAnalyzerGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
