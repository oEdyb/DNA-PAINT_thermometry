# DNA-PAINT Thermometry Analysis

Analysis pipeline for DNA-PAINT thermometry experiments.

## How to Use

1. Download or clone this repository
2. Double-click `start_thermometry_gui.bat`
3. The batch file will automatically:
   - Check if Python is installed
   - Create a virtual environment
   - Install required packages
   - Test dependencies
   - Start the analysis GUI

## Requirements

- Windows operating system
- Python 3.10 or newer
- Internet connection for package installation

## What it does

The pipeline processes DNA-PAINT data through four steps:
1. Extract data from HDF5 files
2. Process localization data and detect binding sites
3. Calculate binding kinetics (on/off times)
4. Estimate binding times using statistical fitting

## Output

Results are saved in organized folders with:
- Binding time data files
- Super-resolution images
- Analysis plots and histograms 