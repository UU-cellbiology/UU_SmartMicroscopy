# **Modular Platform for Smart Microscopy**

Welcome to the **UU Smart Microscopy** repository! This Python-based project is designed to automate and enhance microscope capabilities for smart microscopy experiments, featuring real-time feedback control and modular extensibility.

---

## **Overview**

This repository provides a **library** for controlling microscopes, along with **use-case examples** demonstrating its functionality. The new structure separates core functionality from application-specific implementations, making it easier to integrate and extend.

### Key Features:
- **Library Structure**: Installable via `pip` for easy integration into other projects.
- **GUI Integration**: User-friendly graphical interface built with Tkinter.
- **Real-time Feedback Control**: Supports PID control and custom algorithms.
- **Modular Design**: Easily extendable for custom models and hardware.
- **Demo Mode**: Preloaded datasets for testing and simulation.
- **Multi-Cell Models**: Expanded functionality for handling multiple cells for both nucleus concentration and cell migration.

---

## **Repository Structure**

The repository is organized as follows:

```
UU_smart_microscopy/    # Core library (pip-installable)
│
├── setup.py            # Installation script (includes dependencies)
├── usecases/           # Example applications
│   ├── OutcomeDriven/  # Original main.py logic (renamed)
│   └── Minimal/        # Simplified demo use case
│
├── info/               # Dependencies details and demo data
│
├── microscopeBridge/   # Microscope interfaces
│   ├── micromanager.py
│   └── demo.py
│
├── models/             # Smart microscopy models
│   ├── AutomaticPath.py
│   ├── PID_LEXY_SAM.py
│   └── demo.py
│
├── Interface/          # User interface
│   └── GUI_tkinter.py
│
├── segmentation/       # Segmentation methods
│   ├── SAM.py
│   └── Threshold.py
│
├── Controllers/        # Controller methods
│   ├── PID_controller.py
│   └── Direction_controller.py
│
└── configs/            # Utilities and configurations
    ├── functions.py
    └── globVars.py
```

---

## **Getting Started**

### Prerequisites
- **Python 3.9+** (Tested with 3.9.13)
- Dependencies are automatically installed via `setup.py`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/UU-cellbiology/UU_SmartMicroscopy.git
   cd UU_SmartMicroscopy
   ```

2. Install the library:
   ```bash
   pip install .  # or "pip install setup.py"
   ```

3. [optional] Move to the demo use-case folder
   ```bash
   cd useCases/Minimal
   ```

---

## **Usage**

### **1. Running Use Cases**
Two example applications are provided in `usecases/`:

#### **Minimal Demo**
```bash
python usecases/Minimal/Minimal.py
```
- Quick-start test of basic functionality
- Uses simulated data
- No GPU needed, runs in standard desktop computer

#### **Outcome-Driven Experiment (Original Main)**
```bash
python usecases/OutcomeDriven/OutcomeDriven.py
```
- Full-featured version with GUI
- Configure via `inputs.yaml` (see below)

---

### **2. Configuration (inputs.yaml)**
For the outcome-driven use case, configure `inputs.yaml` with:

#### **General Settings**
```yaml
file_name: "experiment_data"
folder_name: "results"
model: "PID_LEXY_SAM"  # or "AutomaticPath"
microscope: "demo"     # or "micromanager"
n_time_points: 100
time_interval_s: 5
```

#### **Model-Specific Parameters**
- **PID_LEXY_SAM**:
  ```yaml
  PID_coef: [0.5, 0.1, 0.01]
  LEXY_control_parameter: "nucleus_intensity"
  LEXY_normalization_parameters: [min, max]
  ```
  
- **AutomaticPath**:
  ```yaml
  path_type: "circle"
  path_pos: [[500, 500], 100, 20]  # [center, radius, points]
  ```

---

### **3. Running the Program**
#### **a. Starting the GUI**
Launch the application using:
```bash
python main.py
```
The initialization is expected to be in the order of seconds. If SAM segmentation is used for the first time, then dat awill be downloaded and initialization is expected to be in the order of minutes.

#### **b. Interacting with the GUI**
The GUI provides an intuitive interface for controlling the experiment. Key features include:
- **View Selection**: Toggle between the raw camera feed, illumination patterns, or segmented cells.
- **Calibration**: Set and acquire calibration images for modulator-to-camera mapping.
- **Real-Time Visualization**: Monitor segmented cells and illumination patterns live.
- **Acquisition Control**:
  - Start: Begins the acquisition process.
  - Abort: Stops the experiment at any point.

---

## **Extending the Platform**

### **Creating New Use Cases**
1. Make a new directory in `usecases/`
2. Import required modules:
   ```python
   from smart_microscopy import models, microscopeBridge
   ```
3. Implement your experiment logic

### **Adding Models/Microscopes**
Follow the existing patterns in:
- `models/` for new control algorithms
- `microscopeBridge/` for new hardware interfaces

---

## **Contributing**

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Description"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## **Authors & Contact**
- **Alfredo Rates**, **Josiah Passmore**, **Ihor Smal**, **Menno van Laarhoven**, **Jakob Schröder**
- Contact: [a.ratessoriano@uu.nl](mailto:a.ratessoriano@uu.nl)

---
