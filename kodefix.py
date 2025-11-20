from controller import Robot, Keyboard
import numpy as np
import math
from scipy.optimize import minimize_scalar
import scipy.linalg

config = {
    "kinematic": {
        "thigh_length": 0.08,
        "calf_length": 0.09,
        "ankle_length": 0.07,
        "leg_length": 0.24
    },
    "offset": {
        "x_offset": 0.0,
        "y_offset": 0.05,
        "z_offset": 0.0,
        "lx_offset": 0.0,
        "ly_offset": 0.037,
        "lz_offset": -0.24,
        "rx_offset": 0.0,
        "ry_offset": 0.037,
        "rz_offset": -0.24
    },
    "joints_direction": {
        "leg_upper_r": 1.0,
        "leg_lower_r": -1.0,
        "ankle_r": 1.0,
        "foot_r": -1.0,
        "pelv_yaw_r": 1.0,
        "pelv_roll_r": -1.0,
        "leg_upper_l": 1.0,
        "leg_lower_l": 1.0,
        "ankle_l": 1.0,
        "foot_l": -1.0,
        "pelv_yaw_l": -1.0,
        "pelv_roll_l": -1.0
    },
    "walking": {
        "ssp_time": 0.50,
        "dsp_time": 0.50,
        "x_swing": 0.12,
        "y_swing": 0.08,
        "z_swing": 0.15
    },
    "gains": {
        "Kp_imu_pitch": 0.5,
        "Kd_imu_pitch": 0.05,
        "Kp_imu_roll": 0.5,
        "Kd_imu_roll": 0.01,
        "Kp_ankle": 0.5,
        "Ki_ankle": 0.01,
        "Kd_ankle": 0.006,
        "ankle_integral_max": 0.6
    },
    "joint_limits": {
        "LegUpperL": (-2.8, 2.8), "LegLowerL": (-2.8, 2.8), "AnkleL": (-2.8, 2.8), "FootL": (-2.8, 2.8),
        "PelvYL": (-2.8, 2.8), "PelvL": (-2.8, 2.8),
        "LegUpperR": (-2.8, 2.8), "LegLowerR": (-2.8, 2.8), "AnkleR": (-2.8, 2.8), "FootR": (-2.8, 2.8),
        "PelvYR": (-2.8, 2.8), "PelvR": (-2.8, 2.8)
    }
}

# Load config
#Kinematik
thigh_length = config["kinematic"]["thigh_length"]
calf_length = config["kinematic"]["calf_length"]
ankle_length = config["kinematic"]["ankle_length"]
leg_length = config["kinematic"]["leg_length"]
#offset
x_offset = config["offset"]["x_offset"]
y_offset = config["offset"]["y_offset"]
z_offset = config["offset"]["z_offset"]
lx_offset = config["offset"]["lx_offset"]
ly_offset = config["offset"]["ly_offset"]
lz_offset = config["offset"]["lz_offset"]
rx_offset = config["offset"]["rx_offset"]
ry_offset = config["offset"]["ry_offset"]
rz_offset = config["offset"]["rz_offset"]
#joint direction
joints_direction = {
    "LegUpperR": config["joints_direction"]["leg_upper_r"],
    "LegLowerR": config["joints_direction"]["leg_lower_r"],
    "AnkleR": config["joints_direction"]["ankle_r"],
    "FootR": config["joints_direction"]["foot_r"],
    "PelvYR": config["joints_direction"]["pelv_yaw_r"],
    "PelvR": config["joints_direction"]["pelv_roll_r"],
    "LegUpperL": config["joints_direction"]["leg_upper_l"],
    "LegLowerL": config["joints_direction"]["leg_lower_l"],
    "AnkleL": config["joints_direction"]["ankle_l"],
    "FootL": config["joints_direction"]["foot_l"],
    "PelvYL": config["joints_direction"]["pelv_yaw_l"],
    "PelvL": config["joints_direction"]["pelv_roll_l"]
}
#walking paramater
ssp_time = config["walking"]["ssp_time"]
dsp_time = config["walking"]["dsp_time"]
x_swing = config["walking"]["x_swing"]
y_swing = config["walking"]["y_swing"]
z_swing = config["walking"]["z_swing"]
#gains
Kp_imu_pitch = config["gains"]["Kp_imu_pitch"]
Kd_imu_pitch = config["gains"]["Kd_imu_pitch"]
Kp_imu_roll = config["gains"]["Kp_imu_roll"]
Kd_imu_roll = config["gains"]["Kd_imu_roll"]
Kp_ankle = config["gains"]["Kp_ankle"]
Ki_ankle = config["gains"]["Ki_ankle"]
Kd_ankle = config["gains"]["Kd_ankle"]
ankle_integral_max = config["gains"]["ankle_integral_max"]
joint_limits = config["joint_limits"]

# Robot Init
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0
keyboard = Keyboard()
keyboard.enable(timestep)

motor_names = ["LegUpperL", "LegLowerL", "AnkleL", "FootL", "PelvYL", "PelvL",
               "LegUpperR", "LegLowerR", "AnkleR", "FootR", "PelvYR", "PelvR"]
motors = {name: robot.getDevice(name) for name in motor_names}
initial_joints = {name: 0.0 for name in motor_names}
for m in motors.values():
    m.setPosition(0.0)
    m.setVelocity(1.5)

imu = robot.getDevice('InertialUnit')
gyro = robot.getDevice('Gyro')
force_sensor_l = robot.getDevice('ForceSensorL')
force_sensor_r = robot.getDevice('ForceSensorR')
if imu: imu.enable(timestep)
if gyro: gyro.enable(timestep)
if force_sensor_l: force_sensor_l.enable(timestep)
if force_sensor_r: force_sensor_r.enable(timestep)

# LIPM & PREVIEW CONTROL PARAMS 
g = 9.81
zc = 0.27  # Tinggi CoM (m)

T = dt  # Sampling time
N = 160  # Preview horizon (1.6s)

step_length = 0.1  # Langkah maju (m)
step_width = 0.05  # Lebar langkah (m)

cycle_time = 2 * ssp_time + 2 * dsp_time
support_width = 0.30
MASS = 3.3
omega = math.sqrt(g / zc)

# Matriks LIPM diskrit (Kajita 2003)
A = np.array([[1, T], [g*T/zc, 1]])
B = np.array([[0], [-T*g/zc]])
C = np.array([1.0, 0.0])

# Preview control gain (Riccati)
Q = np.eye(2)  # State cost
R = np.array([[1e-6]])  # Control cost
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)  # Feedback gain
Gi = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P) 

# Closed-loop A matrix and its transpose
A_cl = A - B @ K  # (2, 2)
A_cl_T = A_cl.T   # (2, 2)

# Preview gains
Gx = np.zeros(N)

for i in range(N):
    A_cl_T_i = np.linalg.matrix_power(A_cl_T, i)
    # Scalar preview gains
    Gx[i] = (Gi @ A_cl_T_i @ C.T).item()

# Attitude control (tetap)
I_x = 0.04
I_y = 0.05
k_phi = 8.0
d_phi = 1.2
k_theta = 9.0
d_theta = 1.4

A_c_att = np.array([
    [0, 1, 0, 0],
    [-k_phi / I_x, -d_phi / I_x, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -k_theta / I_y, -d_theta / I_y]
])
B_c_att = np.array([
    [0, 0],
    [1.0 / I_x, 0],
    [0, 0],
    [0, 1.0 / I_y]
])

A_att = np.eye(4) + A_c_att * dt
B_att = B_c_att * dt

Q_att = np.diag([100.0, 5.0, 100.0, 5.0])
R_att = np.diag([0.005, 0.005])

P_att = scipy.linalg.solve_discrete_are(A_att, B_att, Q_att, R_att)
K_att = np.linalg.inv(B_att.T @ P_att @ B_att + R_att) @ (B_att.T @ P_att @ A_att)

TORQUE_TO_ANKLE_ANGLE = 0.02

# STATE (LIPM state [pos, vel])
state_x = np.array([0.0, 0.0])  # CoM x relative to stance
state_y = np.array([0.01, 0.0])  # CoM y relative to stance
x_att_state = np.zeros(4)
com_yaw = 0.0

# IMU filtering
filtered_roll = 0.0
filtered_pitch = 0.0
alpha = 0.80

# Walking state
time_elapsed = 0.0
support_leg = 'both'
step_counter = 0
fallen = False
walking = False
last_toggle = -1.0
toggle_delay = 0.3
last_support_change = 0.0

# Foot world positions
left_foot_x = 0.0
right_foot_x = 0.0

# Ankle PID
ankle_integral_pitch = 0.0
ankle_integral_roll = 0.0

# Logging
time_log = []
com_log = []
zmp_log = []
imu_log = []

emergency_counter = 0
max_emergency_attempts = 3

# HELPERS 
def apply_pose(pose_dict):
    for name, angle in pose_dict.items():
        if name in motors:
            motors[name].setPosition(angle)

def compute_orbital_energy(com_x, com_vx, zc, g, MASS):
    Ek = 0.5 * MASS * com_vx**2
    Ep = 0.5 * MASS * omega**2 * com_x**2
    return Ek - Ep

def check_stability(zmp_x, zmp_y, support_width, com_x, com_y):
    half_width = support_width / 2
    return abs(zmp_x - com_x) <= half_width and abs(zmp_y - com_y) <= half_width * 1.2

def compute_ik_pose_support(com_x, com_y, com_yaw, support_leg):
    hip_angle = -0.1 * com_x - 0.05 * filtered_pitch
    knee_angle = -0.1 - 0.05 * abs(com_x) - 0.02 * abs(filtered_pitch)
    ankle_angle = 0.0 + 0.02 * filtered_roll
    if support_leg == 'left':
        hip_angle += 0.02
    elif support_leg == 'right':
        hip_angle -= 0.02
    return hip_angle, knee_angle, ankle_angle

def compute_swing_trajectory(start_pos, end_pos, T, t):
    if t > T:
        return end_pos
    progress = t / T
    a0 = start_pos[0]
    a1 = 0 #kecepatan awal
    a2 = 0 #percepatan awal
    a3 = 10 * (end_pos[0] - start_pos[0])
    a4 = -15 * (end_pos[0] - start_pos[0])
    a5 = 6 * (end_pos[0] - start_pos[0])
    x = a0 + a1 * progress + a2 * progress**2 + a3 * progress**3 + a4 * progress**4 + a5 * progress**5
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * math.sin(math.pi * progress) #bikin dia ke samping geraknya (warn)
    if progress < 0.8:
        z_peak = start_pos[2] + z_swing
        z = start_pos[2] + 4 * z_swing * progress * (1 - progress)
    else:
        z = start_pos[2] + 0.02 * (1 - progress) / 0.2
    return [x, y, z]

def compute_endpoint(support_leg, time_elapsed, ssp_time, dsp_time, x_swing, y_swing, z_swing, left_foot_x, right_foot_x):
    endpoint_l = np.array([lx_offset + left_foot_x, ly_offset, lz_offset])
    endpoint_r = np.array([rx_offset + right_foot_x, ry_offset, rz_offset])

    current_t = time_elapsed % cycle_time
    x_swing_clamped = min(x_swing, 0.12)

    if support_leg == 'left':
        if current_t < ssp_time:
            progress = current_t / ssp_time
            start_pos_r = [right_foot_x, ry_offset, rz_offset]
            end_pos_r = [right_foot_x + x_swing_clamped, ry_offset - y_swing, rz_offset]
            endpoint_r = compute_swing_trajectory(start_pos_r, end_pos_r, ssp_time, current_t)
    elif support_leg == 'right':
        t_start_swing = ssp_time + dsp_time
        t_end_swing = 2 * ssp_time + dsp_time
        if t_start_swing <= current_t < t_end_swing:
            sub_t = current_t - t_start_swing
            progress = sub_t / ssp_time
            start_pos_l = [left_foot_x, ly_offset, lz_offset]
            end_pos_l = [left_foot_x + x_swing_clamped, ly_offset + y_swing, lz_offset]
            endpoint_l = compute_swing_trajectory(start_pos_l, end_pos_l, ssp_time, sub_t)

    return endpoint_l, endpoint_r

def translation_matrix(translation):
    return np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])

def rotation_matrix(euler):
    roll, pitch, yaw = euler
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, 0],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0],
        [-sp,   cp*sr,            cp*cr,            0],
        [0,     0,                0,                1]
    ])

def compute_inverse_kinematic(index_addition, translation_target, rotation_target, thigh_length, calf_length, ankle_length): 
    translation_target = translation_target.copy() 
    translation_target[2] -= leg_length

    matrix_translation = translation_matrix(translation_target)
    matrix_rotation = rotation_matrix(rotation_target)
    matrix_transformation = matrix_translation @ matrix_rotation

    vector = np.array([
        translation_target[0] + matrix_transformation[0, 2] * ankle_length,
        translation_target[1] + matrix_transformation[1, 2] * ankle_length,
        translation_target[2] + matrix_transformation[2, 2] * ankle_length
    ])

    vector_magnitude = np.linalg.norm(vector)
    try:
        acos_result = math.acos((vector_magnitude**2 - thigh_length**2 - calf_length**2) / (2 * thigh_length * calf_length))
    except ValueError:
        return None
    
    angles_offset = {name: 0.0 for name in motor_names}
    angles_offset["LegLowerR" if index_addition == 0 else "LegLowerL"] = acos_result
    try:
        matrix = np.linalg.inv(matrix_transformation)
    except np.linalg.LinAlgError:
        return None
    k = math.hypot(matrix[1, 3], matrix[2, 3])
    l = math.hypot(matrix[1, 3], matrix[2, 3] - ankle_length)
    m = (k**2 - l**2 - ankle_length**2) / (2 * l * ankle_length) if l != 0 else 0.0
    m = np.clip(m, -1, 1)
    try:
        acos_result = math.acos(m)
    except ValueError:
        return None
    if matrix[1, 3] < 0:
        angles_offset["FootR" if index_addition == 0 else "FootL"] = -acos_result
    else:
        angles_offset["FootR" if index_addition == 0 else "FootL"] = acos_result

    matrix_translation = translation_matrix([0, 0, -ankle_length])
    matrix_rotation = rotation_matrix([angles_offset["FootR" if index_addition == 0 else "FootL"], 0.0, 0.0])
    mat = matrix_translation @ matrix_rotation
    try:
        mat_inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return None
    mat2 = matrix_transformation @ mat_inv
    atan_result = math.atan2(-mat2[0, 1], mat2[1, 1])

    if math.isinf(atan_result):
        return None
    k_val = math.sin(angles_offset["LegLowerR" if index_addition == 0 else "LegLowerL"]) * calf_length
    l_val = -thigh_length - math.cos(angles_offset["LegLowerR" if index_addition == 0 else "LegLowerL"]) * calf_length
    n = math.cos(angles_offset["PelvYR" if index_addition == 0 else "PelvYL"]) * vector[0]
    o = math.sin(angles_offset["PelvYR" if index_addition == 0 else "PelvYL"]) * vector[1]
    m_val = n + o
    o = math.cos(angles_offset["PelvR" if index_addition == 0 else "PelvL"]) * vector[2]
    p = math.sin(angles_offset["PelvYR" if index_addition == 0 else "PelvYL"]) * math.sin(angles_offset["PelvR" if index_addition == 0 else "PelvL"]) * vector[0]
    q = math.cos(angles_offset["PelvYR" if index_addition == 0 else "PelvYL"]) * math.sin(angles_offset["PelvR" if index_addition == 0 else "PelvL"]) * vector[1]
    n = o + p - q
    if (k_val * k_val + l_val * l_val) == 0:
        return None
    s = (k_val * n + l_val * m_val) / (k_val * k_val + l_val * l_val)
    t = (n - k_val * s) / l_val if l_val != 0 else 0.0
    theta_result = atan_result
    atan_result = math.atan2(s, t)

    if math.isinf(atan_result):
        return None
    angles_offset["LegUpperR" if index_addition == 0 else "LegUpperL"] = atan_result
    angles_offset["AnkleR" if index_addition == 0 else "AnkleL"] = theta_result - angles_offset["LegLowerR" if index_addition == 0 else "LegLowerL"] - angles_offset["LegUpperR" if index_addition == 0 else "LegUpperL"]
    if angles_offset:
        for joint in angles_offset:
            min_lim, max_lim = joint_limits.get(joint, (-2.8, 2.8))
            angles_offset[joint] = np.clip(angles_offset[joint], min_lim, max_lim)
    return angles_offset

def generate_com_trajectory(step_counter, time_elapsed, cycle_time, step_length, ssp_time, dsp_time, stance_x, stance_y):
    """Generate desired CoM trajectory RELATIVE to current stance leg"""
    com_ref_x = np.zeros(N)
    com_ref_y = np.zeros(N)

    for i in range(N):
        future_t = time_elapsed + i * dt
        mod_t = future_t % cycle_time
        future_step = step_counter + int(future_t // cycle_time)

        if mod_t < ssp_time:
            # SSP kiri: CoM bergerak menuju kaki kiri
            t_phase = mod_t / ssp_time
            com_ref_x[i] = 0.5 * step_length * t_phase  # relatif terhadap stance
            com_ref_y[i] = step_width * (1 - t_phase)   # relatif terhadap stance
        elif mod_t < ssp_time + dsp_time:
            # DSP: CoM bergerak ke tengah
            t_phase = (mod_t - ssp_time) / dsp_time
            com_ref_x[i] = 0.5 * step_length * t_phase
            com_ref_y[i] = step_width * (1 - t_phase)
        elif mod_t < 2*ssp_time + dsp_time:
            # SSP kanan: CoM bergerak menuju kaki kanan
            t_phase = (mod_t - (ssp_time + dsp_time)) / ssp_time
            com_ref_x[i] = -0.5 * step_length * t_phase  # negatif karena relatif ke stance kanan
            com_ref_y[i] = -step_width * (1 - t_phase)
        else:
            # DSP: CoM bergerak ke tengah
            t_phase = (mod_t - (2*ssp_time + dsp_time)) / dsp_time
            com_ref_x[i] = -0.5 * step_length * t_phase
            com_ref_y[i] = -step_width * (1 - t_phase)

    return com_ref_x, com_ref_y

# Main Loop
print('[INFO] Starting improved MPC/LIPM with stance-relative frame and lateral sway')
while robot.step(timestep) != -1:
    t_now = robot.getTime()
    key = keyboard.getKey()
    if key == ord('S') and (t_now - last_toggle) > toggle_delay:
        walking = not walking
        last_toggle = t_now
        if walking:
            state_x = np.array([0.0, 0.0])
            state_y = np.array([0.01, 0.0])
            x_att_state = np.zeros(4)
            time_elapsed = 0.0
            support_leg = 'both'
            step_counter = 0
            fallen = False
            ankle_integral_pitch = 0.0
            ankle_integral_roll = 0.0
            emergency_counter = 0
            left_foot_x = 0.0
            right_foot_x = 0.0
            print('[INIT] Walking started with lateral sway')
        else:
            print('[STOP] Walking paused')

    if not walking or fallen:
        for m in motors.values():
            m.setPosition(0.0)
        continue

    time_elapsed += dt

    # IMU & Gyro
    if imu and gyro:
        roll, pitch, yaw = imu.getRollPitchYaw()
        wx, wy, wz = gyro.getValues()
        filtered_roll = alpha * (filtered_roll + wx * dt) + (1 - alpha) * roll
        filtered_pitch = alpha * (filtered_pitch + wy * dt) + (1 - alpha) * pitch

    x_att_state[0] = filtered_roll
    x_att_state[1] = wx
    x_att_state[2] = filtered_pitch
    x_att_state[3] = wy

    u_att = -K_att @ x_att_state
    u_att = np.clip(u_att, -5.0, 5.0)
    ankle_offset_roll = TORQUE_TO_ANKLE_ANGLE * u_att[0]
    ankle_offset_pitch = TORQUE_TO_ANKLE_ANGLE * u_att[1]

    # Update support leg
    current_t = time_elapsed % cycle_time
    if current_t < ssp_time:
        new_support_leg = 'left'
    elif current_t < ssp_time + dsp_time:
        new_support_leg = 'both'
    elif current_t < 2*ssp_time + dsp_time:
        new_support_leg = 'right'
    else:
        new_support_leg = 'both'

    if new_support_leg != support_leg:
        support_leg = new_support_leg
        last_support_change = time_elapsed
        if support_leg == 'left':
            right_foot_x = left_foot_x + step_length
        elif support_leg == 'right':
            left_foot_x = right_foot_x + step_length
        step_counter += 1
        print(f"[DEBUG] Support leg changed to: {support_leg} at t={time_elapsed:.2f}s | Left={left_foot_x:.3f}, Right={right_foot_x:.3f}")

    # Tentukan posisi stance
    if support_leg == 'left':
        stance_x = left_foot_x
        stance_y = ly_offset
    elif support_leg == 'right':
        stance_x = right_foot_x
        stance_y = ry_offset
    else:
        stance_x = (left_foot_x + right_foot_x) * 0.5
        stance_y = 0.0

    # Generate reference relative to stance leg
    com_ref_x_seq, com_ref_y_seq = generate_com_trajectory(
        step_counter, time_elapsed, cycle_time, step_length, ssp_time, dsp_time,
        stance_x, stance_y  # 
    )

    # Hitung error antara state saat ini dan trajektori referensi
    error_x = com_ref_x_seq - state_x[0]  # Error posisi CoM x
    error_y = com_ref_y_seq - state_y[0]  # Error posisi CoM y

    # Hitung ZMP yang diperlukan (preview control)
    zmp_desired_x = (-K @ state_x).item() + np.dot(Gx, error_x)
    zmp_desired_y = (-K @ state_y).item() + np.dot(Gx, error_y)  # Gunakan Gx untuk Y juga

    # Clamp ZMP
    zmp_desired_x = np.clip(zmp_desired_x, -0.06, 0.06)
    zmp_desired_y = np.clip(zmp_desired_y, -0.08, 0.08)

    # --- Update state LIPM dengan ZMP aktual ---
    # Dalam model LIPM, ZMP adalah input. Jadi kita gunakan zmp_desired_x sebagai input.
    state_x = A @ state_x + B.flatten() * zmp_desired_x  
    state_y = A @ state_y + B.flatten() * zmp_desired_y

    # Batasi kecepatan CoM untuk mencegah lonjakan
    state_x[1] = np.clip(state_x[1], -0.2, 0.2)
    state_y[1] = np.clip(state_y[1], -0.2, 0.2)

    # World frame
    com_world_x = stance_x + state_x[0]
    com_world_y = stance_y + state_y[0]
    zmp_world_x = stance_x + zmp_desired_x
    zmp_world_y = stance_y + zmp_desired_y

    # Stability check
    force_l = force_sensor_l.getValues()[2] if force_sensor_l else 0
    force_r = force_sensor_r.getValues()[2] if force_sensor_r else 0
    if not check_stability(C @ state_x, C @ state_y, support_width, state_x[0], state_y[0])or(support_leg=='left' and force_l<10)or(support_leg=='right' and force_r<10):
        emergency_counter += 1
        if emergency_counter >= max_emergency_attempts:
            print(f"[EMERGENCY] ZMP out of support or no ground contact after {max_emergency_attempts} attempts")
            walking = False
            apply_pose({n: 0.0 for n in motor_names})
            break
            zmp_desired_x *= 0.3
            state_x[0] -= 0.01 
    else:
        emergency_counter = 0

    # DSP: shift CoM forward
    if support_leg == 'both':
        current_t = time_elapsed % cycle_time
        if ssp_time <= current_t < ssp_time + dsp_time:
            # Gerakkan CoM ke depan secara halus
            t_phase = (current_t - ssp_time) / dsp_time
            state_x[0] += 0.02 * dt * t_phase

    # IK and motor commands
    target_ankle_l = 0.0
    target_ankle_r = 0.0

    if support_leg == 'left':
        hip_pitch_l, knee_pitch_l, ankle_pitch_l = compute_ik_pose_support(
            state_x[0], state_y[0], com_yaw, 'left'
        )
        motors['LegUpperL'].setPosition(hip_pitch_l)
        motors['LegLowerL'].setPosition(knee_pitch_l)
        target_ankle_l = ankle_pitch_l
        endpoint_l, endpoint_r = compute_endpoint(support_leg, time_elapsed, ssp_time, dsp_time, x_swing, y_swing, z_swing, left_foot_x, right_foot_x)
        angles_r = compute_inverse_kinematic(0, endpoint_r, [0, 0, com_yaw], thigh_length, calf_length, ankle_length)
        if angles_r is None:
            angles_r = {name: 0.0 for name in motor_names}
            angles_r['LegUpperR'] = 0.1
        for joint, angle in angles_r.items():
            clipped = np.clip(angle * joints_direction[joint], joint_limits[joint][0], joint_limits[joint][1])
            motors[joint].setPosition(clipped)
    elif support_leg == 'right':
        # Gunakan posisi CoM relatif (state_x[0], state_y[0]) untuk IK
        hip_pitch_r, knee_pitch_r, ankle_pitch_r = compute_ik_pose_support(
            state_x[0], state_y[0], com_yaw, 'right'
        )
        motors['LegUpperR'].setPosition(hip_pitch_r)
        motors['LegLowerR'].setPosition(knee_pitch_r)
        target_ankle_r = ankle_pitch_r
        endpoint_l, endpoint_r = compute_endpoint(support_leg, time_elapsed, ssp_time, dsp_time, x_swing, y_swing, z_swing, left_foot_x, right_foot_x)
        angles_l = compute_inverse_kinematic(1, endpoint_l, [0, 0, com_yaw], thigh_length, calf_length, ankle_length)
        if angles_l is None:
            angles_l = {name: 0.0 for name in motor_names}
            angles_l['LegUpperL'] = 0.1
        for joint, angle in angles_l.items():
            clipped = np.clip(angle * joints_direction[joint], joint_limits[joint][0], joint_limits[joint][1])
            motors[joint].setPosition(clipped)
    else:
        apply_pose({n: 0.0 for n in motor_names})
        motors['LegUpperL'].setPosition(0.04)
        motors['LegUpperR'].setPosition(0.04)
        motors['LegLowerL'].setPosition(-0.08)
        motors['LegLowerR'].setPosition(-0.08)

    # Ankle PID
    ankle_error_pitch = filtered_pitch
    ankle_error_roll = filtered_roll
    ankle_integral_pitch += ankle_error_pitch * dt
    ankle_integral_roll += ankle_error_roll * dt
    ankle_integral_pitch = np.clip(ankle_integral_pitch, -ankle_integral_max, ankle_integral_max)
    ankle_integral_roll = np.clip(ankle_integral_roll, -ankle_integral_max, ankle_integral_max)
    ankle_output_pitch = Kp_ankle * ankle_error_pitch + Ki_ankle * ankle_integral_pitch + Kd_ankle * wy
    ankle_output_roll = Kp_ankle * ankle_error_roll + Ki_ankle * ankle_integral_roll + Kd_ankle * wx

    left_ankle_cmd = target_ankle_l + ankle_output_pitch + ankle_offset_pitch
    right_ankle_cmd = target_ankle_r + ankle_output_pitch + ankle_offset_pitch
    left_foot_roll = ankle_output_roll + ankle_offset_roll
    right_foot_roll = -ankle_output_roll - ankle_offset_roll

    motors['AnkleL'].setPosition(np.clip(left_ankle_cmd, joint_limits['AnkleL'][0], joint_limits['AnkleL'][1]))
    motors['AnkleR'].setPosition(np.clip(right_ankle_cmd, joint_limits['AnkleR'][0], joint_limits['AnkleR'][1]))
    motors['FootL'].setPosition(np.clip(left_foot_roll, joint_limits['FootL'][0], joint_limits['FootL'][1]))
    motors['FootR'].setPosition(np.clip(right_foot_roll, joint_limits['FootR'][0], joint_limits['FootR'][1]))

    # Logging (use world CoM and ZMP for clarity)
    time_log.append(t_now)
    com_log.append((com_world_x, com_world_y, state_x[1], state_y[1]))
    zmp_log.append((zmp_world_x, zmp_world_y))
    imu_log.append((filtered_roll, filtered_pitch))

    if int(t_now) % 2 == 0 and t_now > 0:
        print(f"t={t_now:.1f}s | support={support_leg} | "
            f"COMx={com_world_x:.3f} | ZMPx={zmp_world_x:.3f} | "
            f"COMy={com_world_y:.3f} | ZMPy={zmp_world_y:.3f} | "
            f"roll={filtered_roll:.3f} | pitch={filtered_pitch:.3f}")

# Save logs
try:
    import numpy as _np
    data = _np.column_stack([
        time_log,
        [c[0] for c in com_log],
        [c[1] for c in com_log],
        [c[2] for c in com_log],
        [c[3] for c in com_log],
        [z[0] for z in zmp_log],
        [z[1] for z in zmp_log],
        [i[0] for i in imu_log],
        [i[1] for i in imu_log]
    ])
    _np.savetxt('com_data_fixed.csv', data, delimiter=',', header='time,CoM_x,CoM_y,CoM_vx,CoM_vy,ZMP_x,ZMP_y,roll,pitch', comments='')
    print('[INFO] Data saved to com_data_fixed.csv')
except Exception as e:
    print('[WARN] Failed to save logs:', e)

print('[INFO] Controller terminated')