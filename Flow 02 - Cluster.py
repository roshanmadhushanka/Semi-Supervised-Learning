import h2o
import pandas as pd
from h2o.estimators import H2OKMeansEstimator

# Initialize server
h2o.init()

# Predefined variables
response_column = 'class'

# Import training dataset
input_data = pd.read_csv('train.csv')
del input_data[response_column]
input_frame = h2o.H2OFrame(input_data)
columns = list(input_frame.col_names)

# Define H2O model
model = H2OKMeansEstimator(k=3)
model.train(x=columns, training_frame=input_frame)

h2o.save_model(model=model, path='', force=True)
