import csv
import numpy as np

steps = 50
t = np.linspace(0, np.pi/2, steps)

headers = ['iter', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'ee_x', 'ee_y', 'ee_z', 'err', 'elapsed_ms']

with open("build_gpu/trajectory_manipulator_cylinder.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(steps):
        writer.writerow([
            i,
            t[i],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5
        ])

print("Dummy trajectory CSV generated at build_gpu/trajectory_manipulator_cylinder.csv")
