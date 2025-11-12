# Bipedal Walking Control using LIPM and Preview Control (MPC) in Webots

This project implements a bipedal walking controller based on the **Linear Inverted Pendulum Model (LIPM)** and **Model Predictive Control (MPC)** with preview control, inspired by *Kajita et al. (2003)*.  
The controller predicts and stabilizes the robot’s **Center of Mass (CoM)** and **Zero Moment Point (ZMP)** trajectories for dynamically balanced walking.

## Features
- **LIPM + Preview Control (MPC)** for predictive CoM trajectory planning  
- **IMU-based Attitude Stabilization** for real-time balance correction  
- **Inverse Kinematics** for generating leg motion  
- **ZMP stability analysis** and trajectory tracking  
- **Data Logging & Visualization** of CoM and ZMP paths

## Simulation Environment
- **Simulator:** Webots  
- **Language:** Python  
- **Dependencies:** NumPy, SciPy, Matplotlib, Pandas  

## How to Run
1. Open the world file in Webots.  
2. Start the simulation — press **S** to toggle walking.  
3. After simulation, analyze logged data in `com_data_fixed.csv`.

## Visualization
To visualize CoM and ZMP trajectories:
```bash
python visualize_com_zmp.py
