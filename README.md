# ABM-for-irrigation-efficiency-paradox
An agent-based model to simulate farmers' water-land use dynamics in response to both technical irrigation efficiency improvements and economic policy adjustments.
This repository contains the code for the research paper "Modeling Irrigation Efficiency Paradox from Farmers’ Behaviors". 
If you use this code in your research, please cite the corresponding paper.

## File Structure

### Main Simulation File
- `Sim55Agent.py`: The main file for running the simulation.

### Core Components
- `Agent.py`: Defines the farmer class and specifies agent behaviors.
- `plot_utils.py`: Contains all the plotting functions used in the project.

### Policy Simulations
- `SimAgentPolicyPw.py`: Water price - irrigation efficiency combination scenarios.
- `SimAgentPolicyPc.py`: Crop price - irrigation efficiency combination scenarios.
- `SimAgentPolicyAlpha.py`: Land cost - irrigation efficiency combination scenarios.

### Analysis Tools
- `Sensitivity_analysis.py`: Performs sensitivity analysis for the model.

### Input files
- `sub_crop_params.xlsx`: The input file used for initializing agent parameters.

## Usage
To run the simulation, execute the main file:
```bash
python Sim55Agent.py

