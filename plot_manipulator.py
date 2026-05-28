import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import os

# DH Parameters
dh_a = np.array([0.0, -0.427, -0.357, 0.0, 0.0, 0.0])
dh_d = np.array([0.15, 0.0, 0.0, 0.11, 0.09, 0.09])
dh_alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])

def forward_kinematics(q):
    T_global = []
    T_prev = np.eye(4)
    for i in range(6):
        theta = q[i]
        a = dh_a[i]
        d = dh_d[i]
        alpha = dh_alpha[i]
        
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        
        T_local = np.array([
            [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
            [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0,        s_alpha,            c_alpha,           d],
            [0,        0,                  0,                 1]
        ])
        
        T_curr = T_prev @ T_local
        T_global.append(T_curr)
        T_prev = T_curr
        
    return T_global

def get_joint_positions(q):
    T_global = forward_kinematics(q)
    positions = [np.array([0, 0, 0])] # Base position
    for T in T_global:
        positions.append(T[:3, 3])
    return np.array(positions)

def draw_box(ax, x_range, y_range, z_range):
    # Create the meshgrid for the box surfaces
    xx, yy = np.meshgrid(x_range, y_range)
    # Bottom and Top
    ax.plot_surface(xx, yy, np.full_like(xx, z_range[0]), alpha=0.5, color='r')
    ax.plot_surface(xx, yy, np.full_like(xx, z_range[1]), alpha=0.5, color='r')
    
    # Front and Back
    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_surface(np.full_like(yy, x_range[0]), yy, zz, alpha=0.5, color='r')
    ax.plot_surface(np.full_like(yy, x_range[1]), yy, zz, alpha=0.5, color='r')
    
    # Left and Right
    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_surface(xx, np.full_like(xx, y_range[0]), zz, alpha=0.5, color='r')
    ax.plot_surface(xx, np.full_like(xx, y_range[1]), zz, alpha=0.5, color='r')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*40)
    print("Select MPPI Variant to Plot:")
    print("1: Bi-MPPI (x_init_bi_mppi_log.csv)")
    print("2: Standard MPPI (x_init_mppi_log.csv)")
    print("3: Cluster MPPI (x_init_cluster_log.csv)")
    print("4: Log MPPI (x_init_log_mppi_log.csv)")
    print("="*40)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    match choice:
        case '1':
            filename = "x_init_bi_mppi_log.csv"
        case '2':
            filename = "x_init_mppi_log.csv"
        case '3':
            filename = "x_init_cluster_log.csv"
        case '4':
            filename = "x_init_log_mppi_log.csv"
        case _:
            print("Invalid choice. Defaulting to Bi-MPPI (x_init_bi_mppi_log.csv)")
            filename = "x_init_bi_mppi_log.csv"
            
    csv_path = os.path.join(script_dir, filename)
    print(f"\nLoading trajectory from: {filename}")
    
    q_start = None
    q_target = None
    obstacles = []

    # Parse metadata from comments
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith("# START"):
                    parts = line.strip().split(',')
                    q_start = np.array([float(x) for x in parts[1:]])
                elif line.startswith("# TARGET"):
                    parts = line.strip().split(',')
                    q_target = np.array([float(x) for x in parts[1:]])
                elif line.startswith("# OBS"):
                    parts = line.strip().split(',')
                    # Format: # OBS,xmin,xmax,ymin,ymax,zmin,zmax
                    obs_x = [float(parts[1]), float(parts[2])]
                    obs_y = [float(parts[3]), float(parts[4])]
                    obs_z = [float(parts[5]), float(parts[6])]
                    obstacles.append((obs_x, obs_y, obs_z))
                elif not line.startswith("#"):
                    break # Reached the data section
    except FileNotFoundError:
        print(f"{csv_path} not found.")
        return

    # Fallback default values if metadata is missing (for backward compatibility)
    if q_start is None:
        q_start = np.zeros(6)
    if q_target is None:
        q_target = np.zeros(6)
        q_target[0] = np.pi / 2.0

    # Load trajectory data
    try:
        df = pd.read_csv(csv_path, comment='#')
    except Exception as e:
        print(f"Error reading data: {e}")
        return
        
    # Extract joint angles
    qs = df[['q_0', 'q_1', 'q_2', 'q_3', 'q_4', 'q_5']].values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set axes limits for better visualization
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([0.0, 1.5])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # Add obstacles parsed from the CSV metadata
    for obs in obstacles:
        draw_box(ax, obs[0], obs[1], obs[2])

    pos_start = get_joint_positions(q_start)
    pos_target = get_joint_positions(q_target)

    # Plot static lines for start and target
    ax.plot(pos_start[:, 0], pos_start[:, 1], pos_start[:, 2], 'o--', lw=2, color='gray', alpha=0.7, label='Start Position')
    ax.plot(pos_target[:, 0], pos_target[:, 1], pos_target[:, 2], 'o--', lw=2, color='green', alpha=0.7, label='Target Position')
    ax.legend(loc='upper right')

    # Initialize manipulator visualization elements
    line, = ax.plot([], [], [], 'o-', lw=3, color='b', markersize=8, label='Current Pos')
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        q = qs[frame]
        pos = get_joint_positions(q)
        
        # update line data
        line.set_data(pos[:, 0], pos[:, 1])
        line.set_3d_properties(pos[:, 2])
        
        ax.set_title(f"Manipulator Iteration: {frame}")
        return line,

    # Animate the trajectory
    ani = animation.FuncAnimation(
        fig, update, frames=len(qs), init_func=init, 
        blit=False, interval=100, repeat=True
    )
    
    print("Close the window to terminate.")
    plt.show()

if __name__ == '__main__':
    main()
