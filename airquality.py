import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import Sequential
from keras import Dense
from keras_tuner import RandomSearch


#load dataset
data=pd.read_excel("Real_Combine.xlsx")
