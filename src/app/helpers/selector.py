from glob import glob
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys


def select_callback(eclick, erelease):
    global x1, y1, x2, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata


def save_box(event):
    global counter
    counter += 1
    if(event.key == 'enter'):
        file.write(f'{counter},{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n')
        print("Saved")
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=(0, 1, 0), linewidth=1)
        ax.add_patch(rect)
        fig.canvas.draw()

fig = plt.figure(constrained_layout=True)
ax = fig.subplots(1)
selector = RectangleSelector(ax, select_callback, useblit=True,
                             button=[1, 3], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
fig.canvas.mpl_connect('key_press_event', save_box)

img_path = sys.argv[1]
slots_path = sys.argv[2]

counter = 0

img = plt.imread(img_path)
ax.imshow(img)

file = open(slots_path, 'w')

plt.show()

file.close()
