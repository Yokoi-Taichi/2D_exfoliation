import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from sklearn.cluster import KMeans
import PySimpleGUI as sg

import figmatch

LAYOUT = [[sg.Text("Hello World!")],
          [sg.Input("hogehoge",key="-filebrowse-"),
           sg.FileBrowse("Browse", key="-filebrowse-"), sg.Submit()]]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    window = sg.Window("my window", LAYOUT)

    while True:
        event, value = window.read()

        if event is None:
            break
        if event == "-filebrowse-":
            print(value)

    window.close()