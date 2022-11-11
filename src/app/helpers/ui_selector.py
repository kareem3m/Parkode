from glob import glob
from matplotlib.widgets import PolygonSelector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys


def select_callback(verts):
    global x1, y1, x2, y2, x3, y3, x4, y4
    x1, y1 = verts[0]
    x2, y2 = verts[1]
    x3, y3 = verts[2]
    x4, y4 = verts[3]


def save_box(event):
    if(event.key == 'enter'):
        file.write(
            f'{int(x1)},{int(y1)},{int(x2)},{int(y2)},{int(x3)},{int(y3)},{int(x4)},{int(y4)}\n')
        print("Saved")
        rect = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                       fill=False, edgecolor=(0, 1, 0), linewidth=1)
        ax.add_patch(rect)
        fig.canvas.draw()


fig = plt.figure(constrained_layout=True)
ax = fig.subplots(1)
selector = PolygonSelector(ax, select_callback, useblit=True)
fig.canvas.mpl_connect('key_press_event', save_box)

img_path = sys.argv[1]

img = plt.imread(img_path)
ax.imshow(img)

file = open('./ui_spots.csv', 'w')

plt.show()

file.close()
