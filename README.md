# GRNsuite 🍯🐝👅⚡  
**A Python package for analyzing insect taste electrophysiology recordings**  

GRNsuite is an open-source software package designed to process, analyze, and visualize extracellular recordings from insect **gustatory receptor neurons (GRNs)**. It provides both interactive and automated pipelines for electrophysiological data analysis.

**Author:** Rachel H. Parkinson

---

## 🚀 **Current Features**
- **Data Processing**:
  - Automated contact artifact detection
  - Bandpass filtering (100-1000 Hz)
  - Noise reduction
  - Configurable time window selection
- **Spike Detection**:
  - Schmidt trigger detection with configurable thresholds
  - Interactive threshold adjustment
  - Automated vs manual detection comparison
- **Waveform Analysis**:
  - Spike waveform extraction
  - Waveform averaging and comparison
- **Metadata Management**:
  - Automated metadata extraction from filenames
  - Experiment parameter tracking
  - JSON metadata storage

---

## 🛠️ **Installation**
```bash
git clone https://github.com/rachelparkinson/GRNsuite.git
cd GRNsuite
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

---
## 💻 **Usage**

### **Configuration**
Create or modify `parameters.yaml` to set your analysis parameters:
```yaml
# Data acquisition parameters
sampling_rate: 30000  # Hz
offset_time: 0.1     # seconds
analysis_length: 2.0  # seconds

# Spike detection parameters
schmidt_t1: 0.75     # Lower threshold multiplier
schmidt_t2: 1.0      # Upper threshold multiplier

# Processing mode
process_mode: "all"  # Options: "all" or "selected"
selected_samples:    # Used when process_mode is "selected"
    - "20231103-M04-sucr-100-Gal-A1-02"

# Metadata parsing
filename_parsing:
    separator: "-"   # Options: "-" or "_"
    fields:
        - date
        - animal_id
        - stimulus
        - concentration
        - location
        - sensillum_id
        - replicate
```

### **Interactive Workflow**
Use the Jupyter notebook `demo_workflows.ipynb` for:
- Manual contact artifact selection
- Interactive threshold adjustment
- Visual comparison of detection methods
- Single or batch file processing with user confirmation

### **Automated Workflow**
For batch processing using Snakemake:
```bash
# Process all files in data directory
snakemake --cores 1 -s workflow/Snakefile --latency-wait 30

# Use multiple cores for faster processing
snakemake --cores 4 -s workflow/Snakefile --latency-wait 30
```

### **Output Structure**
For each processed file, GRNsuite creates:
```
results/
└── {filename}/
    ├── processed_data.csv   # Filtered and processed signal
    ├── detected_spikes.csv  # Spike times and amplitudes
    ├── waveforms.csv        # Extracted spike waveforms
    └── metadata.json        # Analysis parameters and file metadata
```

---
## 📊 **Current Pipeline Status**
- [x] Basic data loading and preprocessing
- [x] Contact artifact detection
- [x] Signal filtering and noise reduction
- [x] Spike detection (Schmidt trigger)
- [x] Waveform extraction
- [x] Metadata management
- [x] Interactive and automated workflows
- [x] Snakemake integration
- [ ] Unit sorting
- [ ] Response analysis
- [ ] GUI development
- [ ] Advanced visualization tools

---
## 📝 **Contributing**
We welcome contributions! Please see our contributing guidelines for more information.

---
## 📜 **License**
This project is licensed under the MIT License. See LICENSE for details.


