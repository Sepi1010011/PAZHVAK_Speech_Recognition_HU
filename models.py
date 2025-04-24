import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------
# <<< Model Running Func >>>
# --------------------------
