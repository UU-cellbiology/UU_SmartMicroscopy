### README for the Smart Microscopy Repository

---

# **Modular platform for Smart Microscopy**

Welcome to the **UU Smart Microscopy** repository! This Python-based project is designed to automate and enhance the capabilities of microscopes, specifically for smart microscopy experiments with on-the-fly acquisition modifications.

---

## **Overview**

This repository contains the codebase for controlling microscopes based on Python.

### Key Features:
- **GUI Integration**: A user-friendly graphical interface built with Tkinter.
- **Real-time Feedback Control**: Implements PID control and custom algorithms for precision adjustments.
- **Modular Design**: Easily extendable architecture for custom models and hardware setups.
- **Simulated Demo Mode**: A preloaded demo dataset for simulation and testing.

---

## **Contents**

This repository is structured as follows:

1. **Main Files**
   - `main.py`: Core script to initialize and control the microscope via a GUI.
   - `inputs.yaml`: Configuration file for general and advanced settings.

2. **Microscope Bridges**
   - `microscopeBridge/micromanager.py`: Integration with the Micro-Manager software.
   - `microscopeBridge/demo.py`: Demo mode using simulated image stacks.

3. **Models**
   - `models/PID_controller.py`: Implements a PID controller for fluorescence experiments.
   - `models/Direction_controller.py`: Manages directional illumination control for cell migration.

4. **Interfaces**
   - `Interface/GUI_tkinter.py`: Interactive graphical interface for running experiments and controlling settings.

5. **Segmentation**
   - `segmentation/SAM.py`: ...
   - - `segmentation/Threshold.py`: ...

6. **Configurations**
   - `configs/functions.py`: Utility functions for image processing and calculations.
   - `configs/globVars.py`: Global variables for inter-thread communication and shared resources.
---

## **Getting Started**

### Prerequisites
- **Python 3.9.13**
- Required Libraries:
  - `numpy`
  - `tkinter`
  - `scikit-image`
  - `pycromanager` (for integration with Micro-Manager)
  - `tifffile`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/UU-cellbiology/FeedbackMicroscopy.git
   cd FeedbackMicroscopy
   ```

2. Install dependencies:
   -  Install libraries described in .\_info. Tested with Python version 3.9.13.
   -  Installation time expected to be in the order of minutes.
   

4. Configure your setup:
   - Edit the `inputs.yaml` file to match your experiment and hardware.

---

## **Usage**

### **1. Setting Up the Configuration**

Before running the program, ensure that the `inputs.yaml` file is properly configured to suit your experimental needs. Below are some of the key parameters and their descriptions:

#### **General Inputs**
- `file_name`: The name for the output file where data will be saved.
- `folder_name`: The directory where data will be stored.
- `model`: Specify the experiment model (e.g., `'AutomaticPath'` for migration or `'PID_LEXY_SAM'` for fluorescence control).
- `microscope`: Specify the microscope interface (`'micromanager'` for live hardware or `'demo'` for simulation).
- `n_time_points`: The number of time points to acquire.
- `time_interval_s`: The time interval (in seconds) between acquisitions.
- `channels`: Define fluorescence channels and their respective exposure times.

#### **Functionalities**
Each model supports specific functionalities:
- **AutomaticPath**:
  - `path_type`: Shape of the illumination path (`circle`, `square`, etc.).
  - `path_pos`: Position, size, and number of points for the illumination path.
- **PID_LEXY_SAM**:
  - `PID_coef`: PID coefficients for feedback control (Proportional, Integral, Derivative).
  - `LEXY_control_parameter`: Control parameter (`'nucleus_intensity'`, `'cytosol_intensity'`, etc.).
  - `LEXY_normalization_parameters`: Minimum and maximum intensity for normalization.

---

### **2. Running the Program**

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

### **3. Modes of Operation**

#### **a. Demo Mode**
This mode simulates microscope functionality using a preloaded image stack.
1. Set `microscope` to `'demo'` in `inputs.yaml`.
2. Specify the path to the demo data stack in `demo_path`.
3. Run the program to simulate a complete experiment workflow.

#### **b. Real Microscope Mode**
This mode interacts directly with a microscope controlled via Micro-Manager.
1. Ensure that Micro-Manager is installed and configured with your microscope.
2. Set `microscope` to `'micromanager'` in `inputs.yaml`.
3. Run the program to control the microscope hardware.

---

### **4. Extending the Codebase**

#### **Adding New Models**
To create a new model:
1. Create a new Python file in the `models` directory.
2. Implement a class inheriting from the base feedback model.
3. Define the necessary methods for image processing, control logic, and data export.

#### **Integrating New Microscopes**
To support a new microscope system:
1. Implement a new class inheriting from `abstract_bridge` in the `microscopeBridge` directory.
2. Define methods for live image acquisition, modulator control, and shutdown.

---

### **5. Example Workflow**
Below is an example workflow using the **AutomaticPath** model:

1. Configure `inputs.yaml`:
   ```yaml
   microscope: 'micromanager'
   model: 'AutomaticPath'
   path_type: 'circle'
   path_pos: [[500, 500], 100, 20]
   n_time_points: 100
   time_interval_s: 5
   channels:
     - ['GFP', 50]
   ```
2. Run the program:
   ```bash
   python main.py
   ```
3. Use the GUI to:
   - Start the experiment.
   - Visualize illumination patterns and segmented cells in real time.
   - Save data automatically to the specified folder.

---

## **Contributing**

Contributions are welcome! Please follow the guidelines below:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a Pull Request.

---

## **Authors and Contact**

- **Alfredo Rates**, **Josiah Passmore**, **Ihor Smal**, **Menno van Laarhoven**, **Jakob Schröder**
- Contact: [a.ratessoriano@uu.nl](mailto:a.ratessoriano@uu.nl)

---
