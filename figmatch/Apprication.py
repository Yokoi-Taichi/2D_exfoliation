import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from sklearn.cluster import KMeans
import PySimpleGUI as sg

import figmatch as fm

LAYOUT = [[sg.Text("Enter Before image.")],
          [sg.Input("Image before"),
           sg.FileBrowse("Browse", key="-BeforePath-"), sg.Submit(key="-submit-")]]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    window = sg.Window("my window", LAYOUT)

    while True:
        event, value = window.read()

        if event is None:
            break
        if event == "-submit-":
            before_path = value["-BeforePath-"]
            before = cv2.imread(before_path)
            fm.figshow(before)

    window.close()
