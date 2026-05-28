with open('parking.txt', 'w') as f:
    for i in range(30):
        f.write(' '.join(['0.00000e+00'] * 50) + '\n')
