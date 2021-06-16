import sys
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# Hello, this is a final attempt to push vessel recogition to the training_set branch

def load_data():
    """
    Check terminal call usage and load dataset into pandas dataframe.

    No inputs.

    Outputs:
        df           pandas dataframe of stock price history.
    """

    # Check usage
    if len(sys.argv) != 2:
        print("Usage: python3 forecasting.py dataset.csv")
        sys.exit(1)
    else:
        dataset = sys.argv[1]

    # Read dataset into pandas dataframe and print head of dataframe
    df = pd.read_csv(dataset)
    print("\n", df.head(5), "\n")

    return df
















if __name__ == '__main__':

    # print data and show that
    df = load_data()

    final_function()
