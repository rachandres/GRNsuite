# GRNsuite 🍯🐝👅⚡  
**A Python package for analyzing insect taste electrophysiology recordings**  

GRNsuite is an open-source software package designed to process, analyze, and visualize extracellular recordings from insect **gustatory receptor neurons (GRNs)**. It provides an **end-to-end pipeline** for electrophysiological data analysis, from raw signal preprocessing to advanced spike detection, unit sorting, and response characterization.  

---

## 🚀 **Features**
- **Preprocessing**: Noise filtering, movement artifact removal, baseline correction  
- **Spike Detection & Sorting**: Unsupervised clustering for multi-unit separation  
- **Response Analysis**:  
  - Firing rates, burst dynamics, and adaptation patterns  
  - Comparative analysis across stimuli and concentrations  
  - Machine learning-based stimulus-response classification  
- **Visualization**:  
  - Raster plots, peristimulus time histograms (PSTH), and firing rate trends  
  - Heatmaps, response concentration gradients, and interactive plots  
- **Workflows**:  
  - **Command-line interface (CLI)** for batch processing  
  - **Graphical User Interface (GUI)** for non-programmers (coming soon)  
  - **Snakemake integration** for reproducible analysis pipelines  

---

## 🛠️ **Installation**
GRNalyzer is under active development. The package will be available via `pip` and `conda` soon.  

### **Clone & Install from Source**
```bash
git clone https://github.com/yourusername/GRNalyzer.git
cd GRNalyzer
pip install -e .
```

### **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

---
## 💻**Usage**

### **Command-Line Interface (CLI)**
After installation, you can use the CLI for fast analysis:
```bash
grnsuite analyze data/sample.dat --output results.csv
```

### **Python API**
For more customization, use GRNsuite within a Python script:
```python
from grnalyzer import preprocessing, spike_detection
data = preprocessing.load_data("data/sample.dat")
spikes = spike_detection.detect_spikes(data)
print(f"Detected {len(spikes)} spikes")
```

### **Snakemake Workflow**
For batch processing, use Snakemake:
```bash
snakemake -s workflows/Snakefile --cores 4
```

---
## 📊**Visualization**
To generate response plots:
```python
from grnalyzer import visualization
visualization.plot_raster(spikes, title="GRN Response")
```

## 📖**Documentation**
Full documentation, API references, and tutorials will be available at:
COMING SOON

## 📝 **To-Do List**
- [x] Set up repository & package structure  
- [ ] Implement preprocessing & spike detection  
- [ ] Develop GUI for non-programmers 🖥️  
- [ ] Add machine learning-based response classification 🤖  
- [ ] Publish to PyPI and Conda 📦  

---

## 👥 **Contributing**
We welcome contributions! To contribute to **GRNsuite**, follow these steps:  

1. **Fork the repository** to your own GitHub account.  
2. **Create a new branch** for your feature or bug fix:  
   ```bash
   git checkout -b feature-name
   ```
3. **Make your changes** and commit them:
   ```bash
   git commit -m "Added new feature"
   ```
4. **Push your branch** on GitHub:
   ```bash
   git push origin feature-name
   ```
5. **Submit a Pull Request** and describe the changes you've made.

For major changes, please open an **Issue** first to discuss your ideas.

---
## 📜**License**
This project is licensed under the MIT License. See LICENSE for details.


