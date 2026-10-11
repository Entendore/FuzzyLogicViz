import sys
import json
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QGroupBox, QTabWidget, QComboBox, QFormLayout,
    QScrollArea, QFrame, QGridLayout, QPushButton, QFileDialog, QStackedWidget,
    QSplitter, QTextEdit
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# ==========================================
# Fuzzy Logic Engine (Mamdani 2-Input)
# ==========================================
def trimf(x, a, b, c):
    a, b, c = sorted([a, b, c])
    if a == b:
        y = np.clip((c - x) / (c - b + 1e-9), 0, 1)
        y[x <= b] = 1.0
    elif b == c:
        y = np.clip((x - a) / (b - a + 1e-9), 0, 1)
        y[x >= b] = 1.0
    else:
        y = np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))
    return y

class FuzzySystem:
    def __init__(self):
        self.temp_range = np.linspace(0, 50, 100)
        self.hum_range = np.linspace(0, 100, 100)
        self.speed_range = np.linspace(0, 100, 200)
        
        self.params = {
            'temp': { 'cold': [0, 0, 20], 'warm': [10, 25, 40], 'hot': [30, 50, 50] },
            'hum': { 'dry': [0, 0, 50], 'normal': [30, 50, 70], 'humid': [60, 100, 100] },
            'speed': { 'slow': [0, 0, 50], 'medium': [20, 50, 80], 'fast': [60, 100, 100] }
        }
        self.rule_matrix = [
            ['slow', 'slow', 'medium'],
            ['medium', 'medium', 'fast'],
            ['fast', 'fast', 'fast']
        ]
        
        self.tnorm = 'min'
        self.snorm = 'max'
        self.surface_cache = None

    def get_mfs(self, var_name):
        ranges = {'temp': self.temp_range, 'hum': self.hum_range, 'speed': self.speed_range}
        x = ranges[var_name]
        return {k: trimf(x, *v) for k, v in self.params[var_name].items()}

    def invalidate_cache(self):
        self.surface_cache = None

    def compute_surface(self, defuzz_method):
        if self.surface_cache is not None:
            return self.surface_cache
            
        T, H = np.meshgrid(self.temp_range[::2], self.hum_range[::2])
        Z = np.zeros_like(T)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                z, _, _, _, _ = self.infer(T[i, j], H[i, j], defuzz_method)
                Z[i, j] = z
        self.surface_cache = (T, H, Z)
        return self.surface_cache

    def infer(self, crisp_temp, crisp_hum, defuzz_method='Centroid'):
        mu_t = {
            'cold': trimf(np.array([crisp_temp]), *self.params['temp']['cold'])[0],
            'warm': trimf(np.array([crisp_temp]), *self.params['temp']['warm'])[0],
            'hot':  trimf(np.array([crisp_temp]), *self.params['temp']['hot'])[0]
        }
        mu_h = {
            'dry': trimf(np.array([crisp_hum]), *self.params['hum']['dry'])[0],
            'normal': trimf(np.array([crisp_hum]), *self.params['hum']['normal'])[0],
            'humid':  trimf(np.array([crisp_hum]), *self.params['hum']['humid'])[0]
        }
        
        temp_keys = ['cold', 'warm', 'hot']
        hum_keys = ['dry', 'normal', 'humid']
        rule_firings = np.zeros(9)
        aggregated = np.zeros_like(self.speed_range)
        
        rule_weights = []
        rule_peaks = []
        speed_mfs = self.get_mfs('speed')

        for i in range(3):
            for j in range(3):
                if self.tnorm == 'min':
                    fire_strength = min(mu_t[temp_keys[i]], mu_h[hum_keys[j]])
                else: 
                    fire_strength = mu_t[temp_keys[i]] * mu_h[hum_keys[j]]
                    
                rule_firings[i*3 + j] = fire_strength
                
                out_label = self.rule_matrix[i][j]
                clipped = np.minimum(speed_mfs[out_label], fire_strength)
                
                if self.snorm == 'max':
                    aggregated = np.maximum(aggregated, clipped)
                else: 
                    aggregated = aggregated + clipped - aggregated * clipped
                
                peak_x = self.speed_range[np.argmax(speed_mfs[out_label])]
                rule_weights.append(fire_strength)
                rule_peaks.append(peak_x)
        
        if np.max(aggregated) == 0:
            crisp_speed = 0
        elif defuzz_method == 'Centroid':
            crisp_speed = np.sum(self.speed_range * aggregated) / np.sum(aggregated)
        elif defuzz_method == 'MOM (Mean of Max)':
            max_val = np.max(aggregated)
            indices = np.where(aggregated >= max_val - 1e-5)[0]
            crisp_speed = np.mean(self.speed_range[indices])
        elif defuzz_method == 'Bisector':
            total_area = np.trapz(aggregated, self.speed_range)
            half_area = total_area / 2.0
            cum_area = np.cumsum((aggregated[:-1] + aggregated[1:]) / 2 * np.diff(self.speed_range))
            idx = np.searchsorted(cum_area, half_area)
            crisp_speed = self.speed_range[min(idx, len(self.speed_range)-1)]
        elif defuzz_method == 'Weighted Average':
            w_sum = sum(rule_weights)
            crisp_speed = sum(np.array(rule_weights) * np.array(rule_peaks)) / w_sum if w_sum > 0 else 0
            
        return crisp_speed, aggregated, mu_t, mu_h, rule_firings

# ==========================================
# PySide6 GUI Application
# ==========================================
class FuzzyVizApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fuzzy Logic Studio Pro: Dynamics & Analytics")
        self.resize(1500, 950)
        
        self.fuzzy = FuzzySystem()
        
        # Main Splitter Layout (Vertical: Top Controls/Plots, Bottom Trace)
        self.splitter_main = QSplitter(Qt.Vertical)
        self.setCentralWidget(self.splitter_main)
        
        # --- Top Splitter (Horizontal: Controls | Plots) ---
        self.splitter_top = QSplitter(Qt.Horizontal)
        
        # --- Left Side: Controls ---
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        self.tabs = QTabWidget()
        
        # ===== Tab 1: Inputs & Rules =====
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        input_group = QGroupBox("Crisp Inputs")
        input_layout = QGridLayout(input_group)
        
        self.temp_label = QLabel("Temp: 25.0 °C")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0); self.temp_slider.setMaximum(500); self.temp_slider.setValue(250)
        self.temp_slider.valueChanged.connect(self.update_visualization)
        
        self.hum_label = QLabel("Humidity: 50.0 %")
        self.hum_slider = QSlider(Qt.Horizontal)
        self.hum_slider.setMinimum(0); self.hum_slider.setMaximum(1000); self.hum_slider.setValue(500)
        self.hum_slider.valueChanged.connect(self.update_visualization)
        
        input_layout.addWidget(QLabel("Temperature:"), 0, 0)
        input_layout.addWidget(self.temp_label, 0, 1)
        input_layout.addWidget(self.temp_slider, 1, 0, 1, 2)
        input_layout.addWidget(QLabel("Humidity:"), 2, 0)
        input_layout.addWidget(self.hum_label, 2, 1)
        input_layout.addWidget(self.hum_slider, 3, 0, 1, 2)
        
        anim_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Start AC Simulation")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_simulation)
        anim_layout.addWidget(self.btn_play)
        input_layout.addLayout(anim_layout, 4, 0, 1, 2)
        tab1_layout.addWidget(input_group)
        
        rules_group = QGroupBox("Rule Matrix (Firing Strength)")
        rules_layout = QGridLayout(rules_group)
        rules_layout.addWidget(QLabel(""), 0, 0)
        for j, h in enumerate(['Dry', 'Normal', 'Humid']):
            lbl = QLabel(h); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet("font-weight: bold; color: #ddd;")
            rules_layout.addWidget(lbl, 0, j+1)
            
        self.rule_frames = []
        self.rule_combos = []
        for i, t in enumerate(['Cold', 'Warm', 'Hot']):
            lbl = QLabel(t); lbl.setStyleSheet("font-weight: bold; color: #ddd;")
            rules_layout.addWidget(lbl, i+1, 0)
            for j in range(3):
                frame = QFrame()
                frame.setStyleSheet("background-color: rgba(255,255,255,0.05); border-radius: 4px;")
                vbox = QVBoxLayout(frame)
                combo = QComboBox()
                combo.addItems(["slow", "medium", "fast"])
                combo.setCurrentText(self.fuzzy.rule_matrix[i][j])
                combo.currentTextChanged.connect(self.update_rules)
                fire_label = QLabel("0.00")
                fire_label.setAlignment(Qt.AlignCenter)
                fire_label.setStyleSheet("color: #aaa; font-size: 10px;")
                vbox.addWidget(combo); vbox.addWidget(fire_label)
                rules_layout.addWidget(frame, i+1, j+1)
                self.rule_combos.append(combo)
                self.rule_frames.append(fire_label)
                
        tab1_layout.addWidget(rules_group)
        
        self.out_label = QLabel("Output Fan Speed: 0.0 RPM")
        self.out_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4af; padding: 15px;")
        self.out_label.setAlignment(Qt.AlignCenter)
        tab1_layout.addWidget(self.out_label)
        tab1_layout.addStretch()
        self.tabs.addTab(tab1, "Inputs & Rules")
        
        # ===== Tab 2: MF Tuning =====
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.var_selector = QComboBox()
        self.var_selector.addItems(["Temperature", "Humidity", "Fan Speed"])
        self.var_selector.currentIndexChanged.connect(self.change_tuning_var)
        tab2_layout.addWidget(QLabel("Select Variable to Tune:"))
        tab2_layout.addWidget(self.var_selector)
        self.mf_stack = QStackedWidget()
        self.mf_sliders = {}
        for var_key in ['temp', 'hum', 'speed']:
            page = QWidget()
            page_layout = QFormLayout(page)
            max_val = 100 if var_key in ['hum', 'speed'] else 50
            for mf_name, params in self.fuzzy.params[var_key].items():
                grp = QGroupBox(f"{mf_name.capitalize()}")
                vlyt = QVBoxLayout(grp)
                for i, p in enumerate(['a', 'b', 'c']):
                    s = QSlider(Qt.Horizontal)
                    s.setMinimum(0); s.setMaximum(max_val); s.setValue(params[i])
                    s.valueChanged.connect(self.update_mf_params)
                    vlyt.addWidget(QLabel(f"Param {p}")); vlyt.addWidget(s)
                    self.mf_sliders[f"{var_key}_{mf_name}_{p}"] = s
                page_layout.addRow(grp)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
            self.mf_stack.addWidget(scroll)
        tab2_layout.addWidget(self.mf_stack)
        self.tabs.addTab(tab2, "MF Tuning")
        
        # ===== Tab 3: Settings & IO =====
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        presets_grp = QGroupBox("Quick Presets")
        p_lay = QHBoxLayout(presets_grp)
        btn_std = QPushButton("Standard"); btn_std.clicked.connect(lambda: self.load_preset('standard'))
        btn_aggr = QPushButton("Aggressive"); btn_aggr.clicked.connect(lambda: self.load_preset('aggressive'))
        btn_eco = QPushButton("Eco Mode"); btn_eco.clicked.connect(lambda: self.load_preset('eco'))
        p_lay.addWidget(btn_std); p_lay.addWidget(btn_aggr); p_lay.addWidget(btn_eco)
        tab3_layout.addWidget(presets_grp)
        
        settings_grp = QGroupBox("Advanced Fuzzy Operators")
        s_lay = QFormLayout(settings_grp)
        self.defuzz_combo = QComboBox()
        self.defuzz_combo.addItems(["Centroid", "MOM (Mean of Max)", "Bisector", "Weighted Average"])
        self.defuzz_combo.currentTextChanged.connect(self.update_settings)
        self.tnorm_combo = QComboBox()
        self.tnorm_combo.addItems(["Min", "Algebraic Product"])
        self.tnorm_combo.currentTextChanged.connect(self.update_settings)
        self.snorm_combo = QComboBox()
        self.snorm_combo.addItems(["Max", "Probabilistic Sum"])
        self.snorm_combo.currentTextChanged.connect(self.update_settings)
        s_lay.addRow("Defuzzification:", self.defuzz_combo)
        s_lay.addRow("AND (T-Norm):", self.tnorm_combo)
        s_lay.addRow("OR (S-Norm):", self.snorm_combo)
        tab3_layout.addWidget(settings_grp)
        
        io_grp = QGroupBox("Save / Load / Export")
        io_lay = QHBoxLayout(io_grp)
        btn_save = QPushButton("Save Config"); btn_save.clicked.connect(self.save_config)
        btn_load = QPushButton("Load Config"); btn_load.clicked.connect(self.load_config)
        btn_img = QPushButton("Export PNG"); btn_img.clicked.connect(self.export_image)
        io_lay.addWidget(btn_save); io_lay.addWidget(btn_load); io_lay.addWidget(btn_img)
        tab3_layout.addWidget(io_grp)
        tab3_layout.addStretch()
        self.tabs.addTab(tab3, "Settings & IO")
        
        controls_layout.addWidget(self.tabs)
        controls_container.setFixedWidth(450)
        
        # --- Right Side: Matplotlib Canvas & Toolbar ---
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(10, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        
        # Add Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # GridSpec Layout
        gs = gridspec.GridSpec(3, 2, figure=self.figure, hspace=0.4, wspace=0.3)
        self.ax_temp = self.figure.add_subplot(gs[0, 0])
        self.ax_hum = self.figure.add_subplot(gs[0, 1])
        self.ax_speed = self.figure.add_subplot(gs[1, 0])
        self.ax_surface = self.figure.add_subplot(gs[1, 1], projection='3d')
        self.ax_rules = self.figure.add_subplot(gs[2, 0])
        self.ax_time = self.figure.add_subplot(gs[2, 1])
        
        self.splitter_top.addWidget(controls_container)
        self.splitter_top.addWidget(plot_widget)
        self.splitter_top.setSizes([450, 1050])
        
        # --- Bottom Splitter: Inference Trace ---
        self.trace_console = QTextEdit()
        self.trace_console.setReadOnly(True)
        self.trace_console.setMaximumHeight(150)
        self.trace_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace;")
        
        self.splitter_main.addWidget(self.splitter_top)
        self.splitter_main.addWidget(self.trace_console)
        self.splitter_main.setSizes([800, 150])
        
        # Simulation Variables
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_simulation_step)
        self.ambient_temp = 35.0
        self.room_temp = 20.0
        self.time_step = 0
        self.history = {'t': [], 'temp': [], 'hum': [], 'speed': []}
        
        self.apply_theme()
        self.change_tuning_var(0)
        self.recompute_surface()
        self.update_visualization()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #d4d4d4; }
            QGroupBox { font-weight: bold; border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 12px; padding: 12px 6px 6px 6px; background-color: #252526; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4af; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background: #252526; border-radius: 6px; }
            QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 8px 16px; border-radius: 4px; margin: 2px; }
            QTabBar::tab:selected { background: #007acc; color: white; }
            QSlider::groove:horizontal { height: 8px; background: #3c3c3c; border-radius: 4px; }
            QSlider::handle:horizontal { background: #007acc; width: 18px; margin: -5px 0; border-radius: 9px; }
            QSlider::handle:horizontal:hover { background: #1aa0ff; }
            QComboBox, QPushButton { padding: 6px 10px; border: 1px solid #3c3c3c; border-radius: 4px; background-color: #3c3c3c; color: white; }
            QPushButton:hover { background-color: #4c4c4c; border-color: #007acc; }
            QPushButton:checked { background-color: #007acc; border-color: #1aa0ff; }
            QSplitter::handle { background-color: #3c3c3c; width: 3px; height: 3px; }
            QTextEdit { border: 1px solid #3c3c3c; }
        """)

    def change_tuning_var(self, idx):
        self.mf_stack.setCurrentIndex(idx)

    def update_rules(self):
        for i in range(3):
            for j in range(3):
                self.fuzzy.rule_matrix[i][j] = self.rule_combos[i*3 + j].currentText()
        self.recompute_surface()
        self.update_visualization()

    def update_settings(self):
        self.fuzzy.tnorm = 'min' if self.tnorm_combo.currentText() == "Min" else 'prod'
        self.fuzzy.snorm = 'max' if self.snorm_combo.currentText() == "Max" else 'probor'
        self.recompute_surface()
        self.update_visualization()

    def update_mf_params(self):
        for var_key in ['temp', 'hum', 'speed']:
            for mf_name in self.fuzzy.params[var_key].keys():
                self.fuzzy.params[var_key][mf_name] = [
                    self.mf_sliders[f"{var_key}_{mf_name}_a"].value(),
                    self.mf_sliders[f"{var_key}_{mf_name}_b"].value(),
                    self.mf_sliders[f"{var_key}_{mf_name}_c"].value()
                ]
        self.recompute_surface()
        self.update_visualization()

    def load_preset(self, name):
        if name == 'standard':
            self.fuzzy.params = {
                'temp': { 'cold': [0, 0, 20], 'warm': [10, 25, 40], 'hot': [30, 50, 50] },
                'hum': { 'dry': [0, 0, 50], 'normal': [30, 50, 70], 'humid': [60, 100, 100] },
                'speed': { 'slow': [0, 0, 50], 'medium': [20, 50, 80], 'fast': [60, 100, 100] }
            }
        elif name == 'aggressive':
             self.fuzzy.params = {
                'temp': { 'cold': [0, 0, 15], 'warm': [5, 20, 35], 'hot': [25, 50, 50] },
                'hum': { 'dry': [0, 0, 40], 'normal': [20, 40, 60], 'humid': [50, 100, 100] },
                'speed': { 'slow': [0, 0, 30], 'medium': [10, 40, 70], 'fast': [50, 100, 100] }
            }
        elif name == 'eco':
            self.fuzzy.params = {
                'temp': { 'cold': [0, 0, 25], 'warm': [15, 30, 45], 'hot': [35, 50, 50] },
                'hum': { 'dry': [0, 0, 60], 'normal': [40, 60, 80], 'humid': [70, 100, 100] },
                'speed': { 'slow': [0, 0, 60], 'medium': [30, 60, 90], 'fast': [80, 100, 100] }
            }
            
        for var_key in ['temp', 'hum', 'speed']:
            for mf_name, params in self.fuzzy.params[var_key].items():
                for i, p in enumerate(['a', 'b', 'c']):
                    self.mf_sliders[f"{var_key}_{mf_name}_{p}"].setValue(params[i])
        self.recompute_surface()
        self.update_visualization()

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Fuzzy Config", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump({'params': self.fuzzy.params, 'rules': self.fuzzy.rule_matrix}, f, indent=4)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Fuzzy Config", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f:
                data = json.load(f)
            self.fuzzy.params = data['params']
            self.fuzzy.rule_matrix = data['rules']
            for var_key in ['temp', 'hum', 'speed']:
                for mf_name, params in self.fuzzy.params[var_key].items():
                    for i, p in enumerate(['a', 'b', 'c']):
                        self.mf_sliders[f"{var_key}_{mf_name}_{p}"].setValue(params[i])
            for i in range(3):
                for j in range(3):
                    self.rule_combos[i*3 + j].setCurrentText(self.fuzzy.rule_matrix[i][j])
            self.recompute_surface()
            self.update_visualization()

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "", "PNG Files (*.png)")
        if path:
            self.figure.savefig(path, dpi=300, facecolor='#1e1e1e')

    def toggle_simulation(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("⏸ Stop AC Simulation")
            self.timer.start(100)
        else:
            self.btn_play.setText("▶ Start AC Simulation")
            self.timer.stop()

    def run_simulation_step(self):
        crisp_speed, _, _, _, _ = self.fuzzy.infer(self.temp_slider.value()/10.0, self.hum_slider.value()/10.0, self.defuzz_combo.currentText())
        
        # Heat dynamics: Room heats towards ambient, fan cools it down.
        # Added random noise to simulate environment disturbances
        heat_factor = 0.5
        cool_factor = 0.2
        noise = np.random.uniform(-0.5, 0.5)
        
        self.room_temp += heat_factor * (self.ambient_temp - self.room_temp) * 0.1
        self.room_temp -= cool_factor * (crisp_speed / 100.0) * 0.1
        self.room_temp += noise
        self.room_temp = np.clip(self.room_temp, 0, 50)
        
        self.temp_slider.setValue(int(self.room_temp * 10))
        
        # Humidity drifts slowly
        h_val = self.hum_slider.value() + np.random.randint(-2, 3)
        self.hum_slider.setValue(np.clip(h_val, 0, 1000))

    def recompute_surface(self):
        self.fuzzy.invalidate_cache()
        defuzz_method = self.defuzz_combo.currentText()
        self.T_surf, self.H_surf, self.Z_surf = self.fuzzy.compute_surface(defuzz_method)

    def update_visualization(self):
        crisp_temp = self.temp_slider.value() / 10.0
        crisp_hum = self.hum_slider.value() / 10.0
        defuzz_method = self.defuzz_combo.currentText()
        
        crisp_speed, aggregated, mu_t, mu_h, firings = self.fuzzy.infer(crisp_temp, crisp_hum, defuzz_method)
        
        self.temp_label.setText(f"Temp: {crisp_temp:.1f} °C")
        self.hum_label.setText(f"Humidity: {crisp_hum:.1f} %")
        self.out_label.setText(f"Output Fan Speed: {crisp_speed:.1f} RPM")
        
        # Update Inference Trace Console
        trace_text = f"<span style='color:#4af'>--- Inference Trace ---</span><br>"
        trace_text += f"<span style='color:#aaa'>Inputs:</span> Temp={crisp_temp:.1f}°C, Hum={crisp_hum:.1f}%<br>"
        trace_text += f"<span style='color:#aaa'>Fuzzify Temp:</span> Cold={mu_t['cold']:.2f}, Warm={mu_t['warm']:.2f}, Hot={mu_t['hot']:.2f}<br>"
        trace_text += f"<span style='color:#aaa'>Fuzzify Hum:</span> Dry={mu_h['dry']:.2f}, Normal={mu_h['normal']:.2f}, Humid={mu_h['humid']:.2f}<br>"
        trace_text += f"<span style='color:#aaa'>Rules Fired (AND={self.tnorm_combo.currentText()}):</span><br>"
        
        temp_keys = ['Cold', 'Warm', 'Hot']
        hum_keys = ['Dry', 'Normal', 'Humid']
        for i in range(3):
            for j in range(3):
                idx = i*3 + j
                if firings[idx] > 0.01:
                    trace_text += f"&nbsp;&nbsp;&nbsp;Rule {idx+1}: IF {temp_keys[i]} AND {hum_keys[j]} THEN {self.fuzzy.rule_matrix[i][j].upper()} <span style='color:#0f0'>(Fire: {firings[idx]:.2f})</span><br>"
        trace_text += f"<span style='color:#aaa'>Aggregate:</span> OR={self.snorm_combo.currentText()}<br>"
        trace_text += f"<span style='color:#ff0'>Defuzzify ({defuzz_method}):</span> Speed = <span style='color:#f44'><b>{crisp_speed:.2f} RPM</b></span>"
        self.trace_console.setHtml(trace_text)
        
        # Record Time Series History if Simulating
        if self.btn_play.isChecked():
            self.time_step += 1
            self.history['t'].append(self.time_step)
            # Scale temp by 2 to fit on 0-100 axis with the others
            self.history['temp'].append(crisp_temp * 2)
            self.history['hum'].append(crisp_hum)
            self.history['speed'].append(crisp_speed)
            
            max_len = 100
            if len(self.history['t']) > max_len:
                for k in self.history: self.history[k].pop(0)
        else:
            self.history = {'t': [], 'temp': [], 'hum': [], 'speed': []}
            self.ax_time.clear()
            
        for i in range(9):
            fire_val = firings[i]
            lbl = self.rule_frames[i]
            lbl.setText(f"{fire_val:.2f}")
            alpha = int(fire_val * 255)
            lbl.setStyleSheet(f"color: rgba(0, 255, 100, {alpha}); font-size: 10px; font-weight: bold;")
        
        plot_bg = '#1e1e1e'
        text_color = '#d4d4d4'
        grid_color = '#3c3c3c'
        
        for ax in [self.ax_temp, self.ax_hum, self.ax_speed, self.ax_rules, self.ax_time]:
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values(): spine.set_color(grid_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

        # Plot 1: Temp
        self.ax_temp.clear(); self.ax_temp.set_facecolor(plot_bg)
        mfs = self.fuzzy.get_mfs('temp')
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['cold'], 'c-', alpha=0.3)
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['warm'], 'y-', alpha=0.3)
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['hot'], 'm-', alpha=0.3)
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['cold'] * mu_t['cold'], 'c-', linewidth=3)
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['warm'] * mu_t['warm'], 'y-', linewidth=3)
        self.ax_temp.plot(self.fuzzy.temp_range, mfs['hot'] * mu_t['hot'], 'm-', linewidth=3)
        self.ax_temp.axvline(x=crisp_temp, color='r', linestyle='--')
        self.ax_temp.set_title("Input 1: Temperature", fontweight='bold')
        self.ax_temp.set_ylim(-0.05, 1.1); self.ax_temp.grid(True, alpha=0.3, color=grid_color)
        
        # Plot 2: Hum
        self.ax_hum.clear(); self.ax_hum.set_facecolor(plot_bg)
        mfs = self.fuzzy.get_mfs('hum')
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['dry'], 'b-', alpha=0.3)
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['normal'], 'g-', alpha=0.3)
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['humid'], 'r-', alpha=0.3)
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['dry'] * mu_h['dry'], 'b-', linewidth=3)
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['normal'] * mu_h['normal'], 'g-', linewidth=3)
        self.ax_hum.plot(self.fuzzy.hum_range, mfs['humid'] * mu_h['humid'], 'r-', linewidth=3)
        self.ax_hum.axvline(x=crisp_hum, color='r', linestyle='--')
        self.ax_hum.set_title("Input 2: Humidity", fontweight='bold')
        self.ax_hum.set_ylim(-0.05, 1.1); self.ax_hum.grid(True, alpha=0.3, color=grid_color)
        
        # Plot 3: Speed
        self.ax_speed.clear(); self.ax_speed.set_facecolor(plot_bg)
        mfs = self.fuzzy.get_mfs('speed')
        self.ax_speed.plot(self.fuzzy.speed_range, mfs['slow'], 'b-', alpha=0.2)
        self.ax_speed.plot(self.fuzzy.speed_range, mfs['medium'], 'g-', alpha=0.2)
        self.ax_speed.plot(self.fuzzy.speed_range, mfs['fast'], 'r-', alpha=0.2)
        self.ax_speed.fill_between(self.fuzzy.speed_range, aggregated, color='m', alpha=0.3)
        self.ax_speed.plot(self.fuzzy.speed_range, aggregated, 'm-', linewidth=2)
        self.ax_speed.axvline(x=crisp_speed, color='r', linestyle='--', label=f'Out: {crisp_speed:.1f}')
        self.ax_speed.set_title(f"Output: Fan Speed ({defuzz_method})", fontweight='bold')
        self.ax_speed.set_ylim(-0.05, 1.1); self.ax_speed.grid(True, alpha=0.3, color=grid_color)
        
        # Plot 4: 3D Surface
        self.ax_surface.clear(); self.ax_surface.set_facecolor(plot_bg)
        surf = self.ax_surface.plot_surface(self.T_surf, self.H_surf, self.Z_surf, cmap='viridis', alpha=0.6, edgecolor='none')
        
        Z_slice = np.zeros_like(self.fuzzy.temp_range)
        for i, t in enumerate(self.fuzzy.temp_range):
            z, _, _, _, _ = self.fuzzy.infer(t, crisp_hum, defuzz_method)
            Z_slice[i] = z
        self.ax_surface.plot(self.fuzzy.temp_range, np.full_like(self.fuzzy.temp_range, crisp_hum), Z_slice, color='r', linewidth=3)
        self.ax_surface.scatter([crisp_temp], [crisp_hum], [crisp_speed], color='r', s=50, edgecolors='white', depthshade=False)
        
        self.ax_surface.set_title("3D Control Surface", fontweight='bold')
        self.ax_surface.set_xlabel("Temp", color=text_color)
        self.ax_surface.set_ylabel("Hum", color=text_color)
        self.ax_surface.set_zlabel("Speed", color=text_color)
        self.ax_surface.tick_params(colors=text_color)
        for axis in [self.ax_surface.xaxis, self.ax_surface.yaxis, self.ax_surface.zaxis]:
            axis.pane.set_facecolor('#2d2d2d')
            axis.pane.set_edgecolor('#3c3c3c')
            axis.label.set_color(text_color)
            
        # Plot 5: Rule Firings
        self.ax_rules.clear(); self.ax_rules.set_facecolor(plot_bg)
        labels = ['C-D', 'C-N', 'C-H', 'W-D', 'W-N', 'W-H', 'H-D', 'H-N', 'H-H']
        colors = ['#007acc' if f < 0.5 else '#e74c3c' for f in firings]
        self.ax_rules.bar(labels, firings, color=colors)
        self.ax_rules.set_title("Rule Firing Strengths", fontweight='bold')
        self.ax_rules.set_ylim(0, 1.1)
        self.ax_rules.grid(True, alpha=0.3, color=grid_color, axis='y')
        
        # Plot 6: Time Series
        if self.history['t']:
            self.ax_time.clear(); self.ax_time.set_facecolor(plot_bg)
            self.ax_time.plot(self.history['t'], self.history['temp'], color='m', label='Temp (x2)')
            self.ax_time.plot(self.history['t'], self.history['hum'], color='y', label='Humidity')
            self.ax_time.plot(self.history['t'], self.history['speed'], color='c', label='Fan Speed')
            self.ax_time.set_title("System Dynamics (Time Series)", fontweight='bold')
            self.ax_time.set_xlabel("Time Step")
            self.ax_time.set_ylim(0, 110)
            self.ax_time.legend(loc='upper right', facecolor='#252526', edgecolor='#3c3c3c', labelcolor=text_color)
            self.ax_time.grid(True, alpha=0.3, color=grid_color)
        
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Segoe UI"); font.setPointSize(10)
    app.setFont(font)
    window = FuzzyVizApp()
    window.show()
    sys.exit(app.exec())