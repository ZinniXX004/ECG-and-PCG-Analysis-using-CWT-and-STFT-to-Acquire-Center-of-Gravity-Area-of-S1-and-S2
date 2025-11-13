# GUI.py (full patched version written by assistant)
"""
Patched GUI for PCG/ECG Analyzer.
- Fixes threshold/CoG plotting shrink/distortion by ensuring scalogram orientation,
  stable colorbar axes (make_axes_locatable), and explicit axis limits.
- Adds controls for min_area (0=auto) and keep_top in Threshold tab.
- Keeps existing functionality: load, Pan-Tompkins, segmentation, CWT selection.
"""
import os
import json
import traceback
import numpy as np
from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local modules - these must exist in your project
from utils_io import try_load_record
from Pan_Tompkins import detect_r_peaks_with_fallback, plot_pt_pipeline
from Segmentation_ECG_to_PCG import segment_one_cycle
from CWT_PCG import compute_cwt, compute_threshold_and_cogs
from Threshold_Plot_CoG import threshold_mask, compute_cog

# Defaults
DEFAULT_FS = 2000
DEFAULT_CWT_FMIN = 20.0
DEFAULT_CWT_FMAX = 500.0
DEFAULT_CWT_NFREQS = 120
DEFAULT_CWT_COLCOUNT = 300
DEFAULT_PASCAL_A0 = 0.0019
DEFAULT_PASCAL_A_STEP = 0.00031

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

class PCGAnalyzerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCG/ECG Analyzer (patched)")
        self.resize(1250, 920)
        self.setStyleSheet(QT_STYLE)

        # --- State variables ---
        self.record_p_signal = None   # ndarray (n_samples, n_channels)
        self.fs = DEFAULT_FS
        self.sig_names = []
        self.ecg_idx = None
        self.pcg_idx = None
        self.r_peaks = np.array([], dtype=int)

        # Computation caches
        self.current_segment = None
        self.segment_bounds = (0, 0)
        self.current_scalogram = None
        self.current_freqs = None
        self.current_times = None
        self.current_cwt_method = None
        self.current_masks = {}
        self.current_cogs = {}

        # Matplotlib image / colorbar handles
        self.cwt_im = None
        self.cwt_cb = None
        self.cwt_cax = None
        self.thr_im = None
        self.thr_cb = None
        self.thr_cax = None
        self.last_thr_contours = []
        self.last_thr_scat = []

        # Build UI
        self._build_ui()
        self._connect_signals()

    # ---------------- UI build ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_v = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # Top controls
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

        # Record label
        self.record_label = QtWidgets.QLabel("No record loaded")
        main_v.addWidget(self.record_label)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        main_v.addWidget(self.tabs, stretch=1)

        # --- Tab 1: Pan-Tompkins ---
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

        self.fig_pt = Figure(figsize=(10, 6))
        self.canvas_pt = FigureCanvas(self.fig_pt)
        self.toolbar_pt = NavigationToolbar(self.canvas_pt, self)
        t1.addWidget(self.toolbar_pt)
        t1.addWidget(self.canvas_pt, stretch=1)
        self.ax_pt_ecg_raw = self.fig_pt.add_subplot(311)
        self.ax_pt_ecg_proc = self.fig_pt.add_subplot(312)
        self.ax_pt_pcg = self.fig_pt.add_subplot(313)
        self.tabs.addTab(self.tab_pt, "Pan-Tompkins")

        # --- Tab 2: Segmentation ---
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

        self.fig_seg = Figure(figsize=(10, 5))
        self.canvas_seg = FigureCanvas(self.fig_seg)
        self.toolbar_seg = NavigationToolbar(self.canvas_seg, self)
        t2.addWidget(self.toolbar_seg)
        t2.addWidget(self.canvas_seg, stretch=1)
        self.ax_seg_ecg = self.fig_seg.add_subplot(211)
        self.ax_seg_pcg = self.fig_seg.add_subplot(212)
        self.tabs.addTab(self.tab_seg, "Segmentation")

        # --- Tab 3: CWT (with backend selector + controls) ---
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

        self.fig_cwt = Figure(figsize=(10, 6))
        self.canvas_cwt = FigureCanvas(self.fig_cwt)
        self.toolbar_cwt = NavigationToolbar(self.canvas_cwt, self)
        t3.addWidget(self.toolbar_cwt)
        t3.addWidget(self.canvas_cwt, stretch=1)
        self.ax_cwt = self.fig_cwt.add_subplot(111)
        self.tabs.addTab(self.tab_cwt, "CWT")

        # --- Tab 4: Threshold & CoG (with new min_area / keep_top controls) ---
        self.tab_thr = QtWidgets.QWidget()
        t4 = QtWidgets.QVBoxLayout(self.tab_thr)

        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel("S1 thr"))
        self.thr_s1 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s1.setMinimum(1); self.thr_s1.setMaximum(99); self.thr_s1.setValue(60); self.thr_s1.setEnabled(False)
        thr_row.addWidget(self.thr_s1); self.thr_s1_label = QtWidgets.QLabel("0.60"); thr_row.addWidget(self.thr_s1_label)
        thr_row.addWidget(QtWidgets.QLabel("S2 thr"))
        self.thr_s2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.thr_s2.setMinimum(1); self.thr_s2.setMaximum(99); self.thr_s2.setValue(10); self.thr_s2.setEnabled(False)
        thr_row.addWidget(self.thr_s2); self.thr_s2_label = QtWidgets.QLabel("0.10"); thr_row.addWidget(self.thr_s2_label)

        # min_area and keep_top controls
        thr_row.addStretch()
        thr_row.addWidget(QtWidgets.QLabel("min_area"))
        self.thr_min_area_spin = QtWidgets.QSpinBox()
        self.thr_min_area_spin.setRange(0, 1000)  # 0 = adaptive (auto)
        self.thr_min_area_spin.setValue(0)
        self.thr_min_area_spin.setMaximumWidth(110)
        self.thr_min_area_spin.setToolTip("Minimum connected-component area in TF pixels. 0 = automatic/adaptive (recommended).")
        thr_row.addWidget(self.thr_min_area_spin)

        thr_row.addWidget(QtWidgets.QLabel("keep_top"))
        self.thr_keep_top_spin = QtWidgets.QSpinBox()
        self.thr_keep_top_spin.setRange(1, 20)
        self.thr_keep_top_spin.setValue(3)
        self.thr_keep_top_spin.setMaximumWidth(80)
        self.thr_keep_top_spin.setToolTip("If no component passes min_area, keep top-N largest components (N = keep_top).")
        thr_row.addWidget(self.thr_keep_top_spin)

        t4.addLayout(thr_row)

        self.fig_thr = Figure(figsize=(10, 6))
        self.canvas_thr = FigureCanvas(self.fig_thr)
        self.toolbar_thr = NavigationToolbar(self.canvas_thr, self)
        t4.addWidget(self.toolbar_thr)
        t4.addWidget(self.canvas_thr, stretch=1)
        self.ax_thr = self.fig_thr.add_subplot(111)
        self.tabs.addTab(self.tab_thr, "Threshold & CoG")

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

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

        # CWT controls
        self.cwt_fmin.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_fmax.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_nfreqs.editingFinished.connect(self._on_cwt_params_changed)
        self.cwt_backend_combo.currentIndexChanged.connect(self._on_cwt_backend_changed)
        self.cwt_use_freqs_target.stateChanged.connect(self._on_cwt_params_changed)
        self.cwt_colcount_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_a0_spin.valueChanged.connect(self._on_cwt_params_changed)
        self.cwt_astep_spin.valueChanged.connect(self._on_cwt_params_changed)

        # Threshold controls
        self.thr_s1.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_s2.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_min_area_spin.valueChanged.connect(self._on_thr_params_changed)
        self.thr_keep_top_spin.valueChanged.connect(self._on_thr_params_changed)

        self.save_btn.clicked.connect(self._on_save_results)

    # ---------------- core helpers ----------------
    def _ensure_scalogram_orientation_and_axes(self, scal, freqs, times):
        """
        Ensure scalogram rows <-> freqs and cols <-> times are consistent.
        Returns (scal2, freqs2, times2)
        - enforces freqs ascending by flipping scal if needed
        - if times length != scal.shape[1], re-create times linspace between given times[0]..times[-1]
        - if freqs length != scal.shape[0], re-create freqs linspace between given freqs[0]..freqs[-1]
        """
        scal2 = np.asarray(scal)
        if scal2.ndim != 2:
            raise ValueError("scalogram must be 2D")
        nrows, ncols = scal2.shape

        # handle freqs
        if freqs is None or len(freqs) == 0:
            freqs = np.linspace(0.0, 1.0, nrows)
        freqs = np.asarray(freqs)
        if freqs.size != nrows:
            f0 = float(freqs[0]) if freqs.size>0 else 0.0
            fend = float(freqs[-1]) if freqs.size>1 else f0 + max(1.0, nrows/10.0)
            freqs = np.linspace(f0, fend, nrows)
        if freqs[0] > freqs[-1]:
            scal2 = scal2[::-1, :]
            freqs = freqs[::-1]

        # handle times
        if times is None or len(times) == 0:
            times = np.linspace(0.0, float(ncols-1)/float(self.fs), ncols)
        if len(times) != ncols:
            t0 = float(times[0]) if len(times)>0 else 0.0
            tend = float(times[-1]) if len(times)>1 else t0 + float(ncols-1)/float(self.fs)
            times = np.linspace(t0, tend, ncols)

        return scal2, freqs, times

    # ---------------- core actions ----------------
    def clear_all(self):
        """Reset internal state and UI (safe for repeated loads)."""
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

        # disable controls
        self.save_btn.setEnabled(False)
        self.swap_btn.setEnabled(False)
        self.pt_detect_btn.setEnabled(False)
        self.show_pt_pipeline_btn.setEnabled(False)
        self.seg_beat_slider.setEnabled(False)
        self.thr_s1.setEnabled(False)
        self.thr_s2.setEnabled(False)

        # remove/clear existing colorbars, contours, scatter artists and extra axes
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
        self.last_thr_scat = []
        self.cwt_im = None
        self.thr_im = None

        # reset labels and axes
        self.record_label.setText("No record loaded")
        self.method_label.setText("CWT method: (not computed)")
        for fig in (self.fig_pt, self.fig_seg, self.fig_cwt, self.fig_thr):
            fig.clf()
        # recreate axes
        self.ax_pt_ecg_raw = self.fig_pt.add_subplot(311); self.ax_pt_ecg_proc = self.fig_pt.add_subplot(312); self.ax_pt_pcg = self.fig_pt.add_subplot(313)
        self.ax_seg_ecg = self.fig_seg.add_subplot(211); self.ax_seg_pcg = self.fig_seg.add_subplot(212)
        self.ax_cwt = self.fig_cwt.add_subplot(111); self.ax_thr = self.fig_thr.add_subplot(111)
        self.canvas_pt.draw_idle(); self.canvas_seg.draw_idle(); self.canvas_cwt.draw_idle(); self.canvas_thr.draw_idle()
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

        self.ax_pt_ecg_proc.clear()
        self.ax_pt_ecg_proc.plot(t, ecg_filt, linewidth=0.6, label='filtered')
        if mwi.max() != 0:
            self.ax_pt_ecg_proc.plot(t, mwi / mwi.max(), label='MWI (norm)', alpha=0.8)
        self.ax_pt_ecg_proc.set_ylabel("Processed")
        self.ax_pt_ecg_proc.legend(fontsize='small')

        self.ax_pt_pcg.clear()
        t_pcg = np.arange(pcg.size) / float(self.fs)
        self.ax_pt_pcg.plot(t_pcg, pcg, linewidth=0.6)
        self.ax_pt_pcg.set_ylabel("PCG (raw)")
        self.ax_pt_pcg.set_xlabel("Time [s]")

        self.canvas_pt.draw_idle()

        # segmentation slider
        n_beats = max(1, len(self.r_peaks) - 1)
        self.seg_beat_slider.setMaximum(max(0, n_beats - 1))
        self.seg_beat_slider.setEnabled(True)
        self.seg_beat_slider.setValue(0)
        self.seg_beat_label.setText("0")

        self._update_segment_and_downstream(0)
        self.status.showMessage(f"Pan-Tompkins: {len(self.r_peaks)} beats detected.")

    def _on_show_pt_pipeline(self):
        if self.record_p_signal is None or self.ecg_idx is None:
            self.status.showMessage("Load record first.")
            return
        ecg = self.record_p_signal[:, self.ecg_idx].astype(float)
        plot_pt_pipeline(ecg, fs=self.fs, r_peaks=self.r_peaks)

    # ---------- Segmentation & downstream ----------
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

        # Segmentation plotting
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

        self.ax_seg_pcg.clear()
        t_seg = np.arange(start, end) / float(self.fs)
        self.ax_seg_pcg.plot(t_seg, seg)
        self.ax_seg_pcg.set_title("PCG segment")
        self.ax_seg_pcg.set_ylabel("PCG")
        self.ax_seg_pcg.set_xlabel("Time [s]")
        self.canvas_seg.draw_idle()

        # compute CWT
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            fmin, fmax, nfreqs = DEFAULT_CWT_FMIN, DEFAULT_CWT_FMAX, DEFAULT_CWT_NFREQS

        backend_sel = str(self.cwt_backend_combo.currentText()).strip().lower()

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
            self.status.showMessage(f"CWT error: {e}")
            print(traceback.format_exc())
            return

        # times_rel may be relative -> convert to absolute using segment start
        start, _ = self.segment_bounds
        times_abs = times_rel + start / float(self.fs) if (times_rel is not None and len(times_rel)>0) else None

        # enforce consistent orientation and shapes
        try:
            scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(scalogram, freqs, times_abs)
        except Exception:
            self.status.showMessage("CWT returned invalid scalogram shape")
            return

        self.current_scalogram = scal2
        self.current_freqs = freqs2
        self.current_times = times2
        self.current_cwt_method = method
        self.method_label.setText(f"CWT method: {method} (backend: {backend_sel})")

        # compute initial masks & cogs using current threshold params
        self._update_masks_and_cogs()

        # plot CWT & Threshold
        self._plot_cwt_tab()
        self.thr_s1.setEnabled(True); self.thr_s2.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status.showMessage(f"Segment updated. CWT method: {method}")
        self._plot_threshold_tab()

    # ---------- CWT plotting (in-place updates) ----------
    def _plot_cwt_tab(self):
        if self.current_scalogram is None or getattr(self.current_scalogram, "size", 0) == 0:
            self.ax_cwt.clear()
            # remove extra axes (old colorbars)
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

        # ensure shapes / orientation
        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(scal, freqs, times)
        except Exception:
            self.ax_cwt.clear()
            self.canvas_cwt.draw_idle()
            return

        self.current_scalogram = scal; self.current_freqs = freqs; self.current_times = times

        # remove old colorbar axes to avoid shrinking main axes
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
        if getattr(self, 'cwt_im', None) is None:
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
        else:
            try:
                self.cwt_im.set_data(scal)
                self.cwt_im.set_extent(extent)
                vmin = np.nanmin(scal); vmax = np.nanmax(scal)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    self.cwt_im.set_clim(vmin, vmax)
            except Exception:
                self.cwt_im = None
                self._plot_cwt_tab()
                return

        try:
            self.ax_cwt.set_xlim(times[0], times[-1])
            self.ax_cwt.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass

        self.ax_cwt.set_xlabel("Time [s]"); self.ax_cwt.set_ylabel("Frequency [Hz]")
        self.ax_cwt.set_title(f"Scalogram ({self.current_cwt_method})")
        self.canvas_cwt.draw_idle()

    # ---------- Threshold & CoG: compute & plot ----------
    def _get_thr_params(self):
        ma = int(self.thr_min_area_spin.value())
        if ma == 0:
            min_area = None
        else:
            min_area = ma
        keep_top = int(self.thr_keep_top_spin.value())
        return min_area, keep_top

    def _update_masks_and_cogs(self):
        """Recompute masks and CoG using current sliders and threshold params."""
        if self.current_scalogram is None:
            return
        s1 = self.thr_s1.value() / 100.0
        s2 = self.thr_s2.value() / 100.0
        min_area, keep_top = self._get_thr_params()
        try:
            mask1 = threshold_mask(self.current_scalogram, s1, min_area=min_area, keep_top=keep_top)
            mask2 = threshold_mask(self.current_scalogram, s2, min_area=min_area, keep_top=keep_top)
            # make sure mask shapes align: expected (n_freqs, n_times)
            if mask1 is not None and mask1.shape != self.current_scalogram.shape:
                if mask1.T.shape == self.current_scalogram.shape:
                    mask1 = mask1.T
            if mask2 is not None and mask2.shape != self.current_scalogram.shape:
                if mask2.T.shape == self.current_scalogram.shape:
                    mask2 = mask2.T

            # compute CoG (pass mask explicitly)
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
        """Plot threshold result. Robust clearing + shape checks to avoid distortions."""
        # clear axis and remove any extra axes (colorbars) first
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

        # remove old artists
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

        # ensure consistent orientation and axes
        try:
            scal, freqs, times = self._ensure_scalogram_orientation_and_axes(self.current_scalogram, self.current_freqs, self.current_times)
        except Exception:
            scal = self.current_scalogram
            freqs = self.current_freqs
            times = self.current_times

        # update stored copies
        self.current_scalogram = scal; self.current_freqs = freqs; self.current_times = times

        extent = [times[0], times[-1], freqs[0], freqs[-1]]

        # create image and colorbar using make_axes_locatable
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

        # overlay masks safely
        mask1 = self.current_masks.get('S1', None)
        mask2 = self.current_masks.get('S2', None)

        def safe_contour(ax, times_arr, freqs_arr, mask_arr, **kwargs):
            if mask_arr is None:
                return None
            mask_arr = np.asarray(mask_arr)
            # try to correct transpose mismatch
            if mask_arr.shape != (len(freqs_arr), len(times_arr)):
                if mask_arr.T.shape == (len(freqs_arr), len(times_arr)):
                    mask_arr = mask_arr.T
                else:
                    # incompatible shape -> skip contour
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

        # overlay CoG markers (if present)
        cog1 = self.current_cogs.get('S1', None)
        cog2 = self.current_cogs.get('S2', None)
        if cog1 is not None:
            try:
                art1 = self.ax_thr.scatter([cog1[0]], [cog1[1]], marker='x', s=80, c='white', zorder=5)
                self.last_thr_scat.append(art1)
            except Exception:
                pass
        if cog2 is not None:
            try:
                art2 = self.ax_thr.scatter([cog2[0]], [cog2[1]], marker='o', s=80, edgecolors='cyan', facecolors='none', zorder=5)
                self.last_thr_scat.append(art2)
            except Exception:
                pass

        # enforce axis limits explicitly to avoid autoscale jumps
        try:
            self.ax_thr.set_xlim(times[0], times[-1])
            self.ax_thr.set_ylim(freqs[0], freqs[-1])
        except Exception:
            pass

        self.ax_thr.set_xlabel("Time [s]"); self.ax_thr.set_ylabel("Frequency [Hz]")
        # update legend safely
        try:
            leg = self.ax_thr.get_legend()
            if leg is not None:
                try: leg.remove()
                except Exception: pass
            self.ax_thr.legend(loc='upper right', fontsize='small')
        except Exception:
            pass

        self.canvas_thr.draw_idle()

    # ---------- handlers for CWT / thr changes ----------
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
        if self.current_segment is None:
            return
        try:
            fmin = float(self.cwt_fmin.text()); fmax = float(self.cwt_fmax.text()); nfreqs = int(self.cwt_nfreqs.text())
        except Exception:
            fmin, fmax, nfreqs = DEFAULT_CWT_FMIN, DEFAULT_CWT_FMAX, DEFAULT_CWT_NFREQS
        self._compute_cwt_for_current_segment(fmin, fmax, nfreqs)

    def _compute_cwt_for_current_segment(self, fmin, fmax, nfreqs):
        if self.current_segment is None:
            return
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

        start, _ = self.segment_bounds
        times_abs = times_rel + start / float(self.fs) if (times_rel is not None and len(times_rel)>0) else None

        try:
            scal2, freqs2, times2 = self._ensure_scalogram_orientation_and_axes(scalogram, freqs, times_abs)
        except Exception:
            self.status.showMessage("CWT returned invalid scalogram shape")
            return

        self.current_scalogram = scal2
        self.current_freqs = freqs2
        self.current_times = times2
        self.current_cwt_method = method
        self.method_label.setText(f"CWT method: {method} (backend: {backend_sel})")

        # recompute masks & CoG using current slider values and updated threshold params
        self._update_masks_and_cogs()
        self._plot_cwt_tab()
        self._plot_threshold_tab()
        self.status.showMessage(f"CWT recomputed ({method}) using backend={backend_sel}")

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

    # ---------- save results ----------
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

# ---------- run helper ----------
def run_app():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = PCGAnalyzerGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()