import numpy as np
import configparser


def read_config(filename, profile):
    config = configparser.ConfigParser()
    config.read(filename)
    media_path = config[profile]['media_path']
    slots_path = config[profile]['slots_path']
    transform_type = config[profile]['transform_type']
    model_type = config[profile]['model_type']
    model_path = config[profile]['model_path']
    return media_path, slots_path, transform_type, model_type, model_path


def read_slots(filename):
    slots = []
    with open(filename, 'r') as f:
        for line in f:
            slot_id, x1, y1, x2, y2 = line.split(',')
            slots.append((int(slot_id), int(x1), int(y1), int(x2), int(y2)))
    return slots


def read_slots_cnr(filename):
    print(f'Reading slots from file: {filename}')
    slots = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            _, x, y, w, h = line.split(',')
            x = int(np.interp(x, [0, 2592], [0, 1000]))
            y = int(np.interp(y, [0, 1944], [0, 750]))
            w = int(np.interp(w, [0, 2592], [0, 1000]))
            h = int(np.interp(h, [0, 1944], [0, 750]))
            slots.append((x, y, x+w, y+h))
    return slots
