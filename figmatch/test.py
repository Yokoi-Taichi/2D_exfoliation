import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
#from matchscale import Cluster, figshow, OneCH2TreeCH, ObjectDetecting

class hoge():
    def __init__(self,a,b):
        self.a= a
        self.b= b
        self.plus=self.a+self.b

class fuga(hoge):
    def __init__(self,c,d,e):
        super().__init__(c,d)
        self.e= e
    
    def sum(self):
        return self.plus

if __name__ == '__main__':
    img = cv2.imread('VHX_000005.jpg')

