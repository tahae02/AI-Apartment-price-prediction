import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Load dataset into original_df
file_path = "C:/Users/tahaa/OneDrive/Desktop/Intro to AI cw/IN3062-Intro-to-AI-Group-14/germany apartments.csv" # Taha laptop
file_path2 = "C:/Users/Taha/OneDrive/Desktop/Intro to AI cw/IN3062-Intro-to-AI-Group-14/germany apartments.csv" # Taha PC
# download the dataset and copy the file path into here, change the pd.read_csv to the respective file path

original_df = pd.read_csv(file_path2)

# create numerical_df with numerical columns 
numerical_df = original_df.copy()
numerical_df = pd.concat([original_df.select_dtypes(include=np.number),
                          # here we incorporated boolean values in the dataframe alongside the numerical values
                          # this was the make the dataframe better represent the dataset, more info on this in the report 
                          # the boolean values are converted to their respective integer representations as it would be easier to process
                          original_df.select_dtypes(include=bool).astype(int)],
                          axis=1)


# function to check numerical_df_info, prints out info about the current state of the dataframe
def numerical_df_info():
    # overview of numerical_df
    print('overview of numerical_df:')
    print(numerical_df, '\n')
    
    # data types of numerical_df
    print('data types of numerical_df:')
    print(numerical_df.dtypes, '\n')
    
    # missing values of numerical_df
    print('missing values of numerical_df:')
    print(numerical_df.isnull().sum(), '\n')


# remove columns with irrelevant features
numerical_df = numerical_df.drop(columns=['scoutId', 'yearConstructedRange', 'baseRentRange', 'geo_plz',
                                          'noRoomsRange', 'livingSpaceRange'])

# remove columns with too many na values
numerical_df = numerical_df.drop(columns=['telekomHybridUploadSpeed', 'noParkSpaces', 'thermalChar',
                                          'heatingCosts', 'lastRefurbish', 'electricityBasePrice',
                                          'electricityKwhPrice'])

# remove rows where there are n/a values
numerical_df = numerical_df.dropna()

# define X, totalRent, baseRent
#defining the dependant and independant variables
X = numerical_df[['serviceCharge', 'picturecount', 'pricetrend', 'telekomUploadSpeed', #'yearConstructed',
                  'livingSpace', 'noRooms', 'floor', 'numberOfFloors',
                  'newlyConst', 'balcony', 'hasKitchen', 'cellar', 'lift', 'garden'
                  ]]
baseRent = numerical_df['baseRent']

# split into training and testing data
X_train, X_test, baseRent_train, baseRent_test = train_test_split(X, baseRent, test_size=0.20, random_state=42)

# standardising the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# neural network model creation begins here
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# specify the learning rate 
lr = 0.001 
# numerical_df_info()

# create an object of the Adam optimiser with the specified learning rate
adam_optimizer = Adam(learning_rate=lr)

# Compile the model with the custom optimiser
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# the following code is to plot a loss/epoch graph for individual epoch iterations for a single epoch value(1/epochs, 2/epochs, etc)

# train the model and save training history
history = model.fit(X_train_scaled, baseRent_train, epochs=300, batch_size=64, validation_split=0.1, verbose=2)

# predict values for X_test
baseRent_pred = model.predict(X_test_scaled)

# flatten the predictions
baseRent_pred = baseRent_pred.flatten()

# output RMSE and Coefficient of determination
rmse = np.sqrt(mean_squared_error(baseRent_test, baseRent_pred))
r2 = r2_score(baseRent_test, baseRent_pred)
print('RMSE =', rmse)
print('Coefficient of determination: %.2f' % r2)

# Plot the training loss over epochs
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# this is used to show the graph with the labelled axis
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


# numerical_df_info()

# the following code is to plot the loss against epoch for a range of epoch values
# this code was useful to find the epoch value which resulted in the lowest END LOSS VALUES
'''
train_loss = []
val_loss = []

start = 250
end = 300

for e in range(start, end, 2):
    # Train the model and save training history
    history = model.fit(X_train_scaled, baseRent_train, epochs=e, batch_size=64, validation_split=0.1, verbose=2)
    
    # Append training and validation loss to lists
    train_loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])

    # Evaluate the model on the test set
    baseRent_pred = model.predict(X_test_scaled)
    baseRent_pred = baseRent_pred.flatten()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(baseRent_test, baseRent_pred))

    # Output training loss and evaluation metrics
    print('\nEpoch {}, Training Loss: {}'.format(e, train_loss[-1]))
    print('Epoch {}, Validation Loss: {}'.format(e, val_loss[-1]))
    print('Epoch {}, RMSE: {}'.format(e, rmse))
    print('Epoch {}, Coefficient of determination: {}'.format(e, r2_score(baseRent_test, baseRent_pred)))
    print("Epoch {}, the epoch value is {}".format(e, e))
    print("")

# Plot the training and validation loss over epochs
plt.plot(range(start, end, 2), train_loss, label='Training Loss')
plt.plot(range(start, end, 2), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
