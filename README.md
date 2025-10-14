# Adaptive Fractionation for PTV-OAR Overlap

This repository contains the complete code for the adaptive fractionation methodology described in our paper. The approach optimizes radiation therapy dose delivery when there is overlap between the Planning Target Volume (PTV) and Organs at Risk (OAR)..

## 🔬 Overview

Adaptive fractionation is an approach that dynamically adjusts radiation dose delivery based on imaging to maximize PTV coverage. This repository provides:

- **Core algorithm implementation**
- **Complete patient data**
- **Dose-volume histogram (DVH) validation for 6 replanned patients**
- **Interactive web interface**
- **Reproduction of Paper Figures** 

## 📁 Repository Structure

### Core Implementation
- **`adaptive_fractionation_overlap/`** - Main Python package
  - `core_adaptfx.py` - Core adaptive fractionation algorithm using dynamic programming
  - `helper_functions.py` - Statistical functions, penalty calculations, and utility methods

### Data and Evaluation
- **`evaluation/`** - Analysis and paper figures
  - `ACTION_patients_overlap_only.xlsx` - Clinical patient overlap data 
  - `illustrations_for_paper.ipynb` - Jupyter notebook generating paper figures and analysis

- **`DVHs/`** - Dose-Volume Histograms for validation
  - Contains DVH data for 6 replanned patients (Patients 3, 14, 18, 25, 26, 49)
  - Each patient folder includes both adaptive fractionation (AF) and original (og) treatment DVHs
  - Data validates actual dose distributions achieved with adaptive fractionation doses

- **`AdaptFract_MasterList_anonymised_SUPPLEMENT.xlsx`** - Supplementary patient data

### Application Interface
- **`app.py`** - Streamlit web application for interactive treatment planning
- **`requirements.txt`** - Python dependencies
- **`setup.py`** - Package installation configuration

## 🚀 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Overlap_adaptfx
```

2. Install the package and dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## 💻 Usage

### Core Algorithm
```python
import adaptive_fractionation_overlap as afx

# Calculate optimal dose for current fraction
policies, policies_overlap, volume_space, physical_dose, penalty_added, \
values, dose_space, probabilities, final_penalty, values_actual_frac, \
dose_space_adapted = afx.adaptive_fractionation_core(
    fraction=2, 
    volumes=patient_overlaps[:3],
    accumulated_dose=6, 
    number_of_fractions=5,
    min_dose=6, 
    max_dose=10, 
    mean_dose=6.6
)

# Calculate complete treatment plan
physical_doses, accumulated_doses, total_penalty = afx.adaptfx_full(
    volumes=patient_overlaps,
    number_of_fractions=5,
    min_dose=6,
    max_dose=10, 
    mean_dose=8
)
```

### Interactive Web Interface
Launch the Streamlit application for treatment planning:
```bash
streamlit run app.py
```

The web interface provides:
- Real-time dose optimization
- Treatment plan precomputation  
- Full plan calculation and analysis
- Interactive visualization of policies and probability distributions

### Paper Analysis
The analysis and figures from our paper can be reproduced using:
```bash
jupyter notebook evaluation/illustrations_for_paper.ipynb
```

## 📚 Citation

If you use this code in your research, please cite our paper:
```
[Citation details to be added]
```

## 📞 Contact

For questions or collaborations:
- **Author**: Yoel Pérez Haas
- **Email**: yoel.perezhaas@usz.ch
- **Institution**: University Hospital Zurich

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This repository contains anonymized clinical data in compliance with patient privacy regulations. The adaptive fractionation methodology is designed for research purposes and requires clinical validation before clinical implementation.
