import numpy as np

c = 9

input_filename = f'E:\gp\CNR-EXT_FULL_IMAGE_1000x750\camera{c}.csv'
output_filename = f'E:\gp\CNR-EXT_FULL_IMAGE_1000x750\scaled_labels\camera{c}.csv'

in_file = open(input_filename, 'r')
out_file = open(output_filename, 'w')

in_file.readline()

for line in in_file:
    id, x, y, w, h = line.split(',')
    x = int(np.interp(x, [0, 2592], [0, 1000]))
    y = int(np.interp(y, [0, 1944], [0, 750]))
    w = int(np.interp(w, [0, 2592], [0, 1000]))
    h = int(np.interp(h, [0, 1944], [0, 750]))
    out_file.write(f'{id},{x},{y},{x+w},{y+h}\n')

in_file.close()
out_file.close()
