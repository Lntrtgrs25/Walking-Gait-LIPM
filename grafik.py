import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Load data from CSV
data = pd.read_csv('com_data_fixed.csv')

# Extract columns
time = data['time'].values
com_x = data['CoM_x'].values
com_y = data['CoM_y'].values
com_vx = data['CoM_vx'].values
com_vy = data['CoM_vy'].values
zmp_x = data['ZMP_x'].values
zmp_y = data['ZMP_y'].values
roll = data['roll'].values
pitch = data['pitch'].values

# Plot 1: ZMPx and CoMx vs time
plt.figure(figsize=(10, 6))
plt.plot(time, zmp_x, label='ZMP_x (m)', color='blue')
plt.plot(time, com_x, label='CoM_x (m)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('ZMP_x and CoM_x vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: ZMPy and CoMy vs time
plt.figure(figsize=(10, 6))
plt.plot(time, zmp_y, label='ZMP_y (m)', color='green')
plt.plot(time, com_y, label='CoM_y (m)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('ZMP_y and CoM_y vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3 : CoM and ZMP Trajectory (Top view)
plt.figure(figsize=(10, 5))
plt.plot(com_x, com_y, color='red', linewidth=2.0, label='CoM Trajectory')
plt.plot(zmp_x, zmp_y, color='blue', linewidth=1.2, marker='x', markersize=4, label='ZMP Trajectory')

plt.title('CoM and ZMP Trajectory (Top View)', fontsize=13)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Approximate foot positions 
left_foot_x_approx = com_x - 0.05  # Rough offset
right_foot_x_approx = com_x + 0.05
com_z = 0.27  # Constant CoM height
foot_z = 0.0  # Ground

# Cycle parameters from controller
ssp_time = 0.5
dsp_time = 0.5
cycle_time = 2 * ssp_time + 2 * dsp_time  # 2.0 s

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(np.min(com_x) - 0.1, np.max(com_x) + 0.1)
ax.set_ylim(-0.1, 0.4)  # Z from ground to CoM height
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Z Position (m)')
ax.set_title('2D Side View Animation: CoM, Feet, and CoM-Feet Links')
ax.grid(True)

# Plot elements
com_point, = ax.plot([], [], 'ro', markersize=8, label='CoM')
left_foot_line, = ax.plot([], [], 'b-', linewidth=4, label='Left Foot')
right_foot_line, = ax.plot([], [], 'g-', linewidth=4, label='Right Foot')
com_left_foot_line, = ax.plot([], [], color='lightblue', linewidth=2, label='CoM-Left Foot Link')  # Biru muda
com_right_foot_line, = ax.plot([], [], color='lightgreen', linewidth=2, label='CoM-Right Foot Link')  # Hijau muda
ax.legend()

def init():
    com_point.set_data([], [])
    left_foot_line.set_data([], [])
    right_foot_line.set_data([], [])
    com_left_foot_line.set_data([], [])
    com_right_foot_line.set_data([], [])
    return com_point, left_foot_line, right_foot_line, com_left_foot_line, com_right_foot_line

def animate(i):
    # Determine support leg based on time % cycle_time
    t_mod = time[i] % cycle_time
    if t_mod < ssp_time:
        support_leg = 'left'  # Left support, right swing
    elif t_mod < ssp_time + dsp_time:
        support_leg = 'both'  # DSP, no swing
    elif t_mod < 2 * ssp_time + dsp_time:
        support_leg = 'right'  # Right support, left swing
    else:
        support_leg = 'both'  # DSP, no swing
    
    # CoM
    com_point.set_data([com_x[i]], [com_z])
    
    # Feet (as horizontal lines at ground level, width based on approx)
    left_foot_line.set_data([left_foot_x_approx[i] - 0.02, left_foot_x_approx[i] + 0.02], [foot_z, foot_z])
    right_foot_line.set_data([right_foot_x_approx[i] - 0.02, right_foot_x_approx[i] + 0.02], [foot_z, foot_z])
    
    # CoM-Feet links
    # CoM to left foot (light blue): thins if left swing (support_leg == 'right')
    left_linewidth = 1 if support_leg == 'right' else 2
    com_left_foot_line.set_data([com_x[i], left_foot_x_approx[i]], [com_z, foot_z])
    com_left_foot_line.set_linewidth(left_linewidth)
    
    # CoM to right foot (light green): thins if right swing (support_leg == 'left')
    right_linewidth = 1 if support_leg == 'left' else 2
    com_right_foot_line.set_data([com_x[i], right_foot_x_approx[i]], [com_z, foot_z])
    com_right_foot_line.set_linewidth(right_linewidth)
    
    return com_point, left_foot_line, right_foot_line, com_left_foot_line, com_right_foot_line

anim = FuncAnimation(fig, animate, init_func=init, frames=len(time), interval=50, blit=True)
plt.show()
