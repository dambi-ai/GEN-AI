import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the dataset (pima_indian_diabetes.csv DataSet available in this same github link)
df = pd.read_csv("https://raw.githubusercontent.com/dambi-ai/GEN-AI/refs/heads/Bronze/pima_indian_diabetes.csv")

# Get all columns except the last one (X)
X = df.iloc[:, :-1]
# Get only the last column for the target variable (y)
y = df.iloc[:, -1]

#Define the neural network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))   # First hidden layer
model.add(Dense(8, activation='relu'))                 # Second hidden layer
model.add(Dense(1, activation='sigmoid'))              # Output layer

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Print summary
print(model.summary())

#Draw the neural network architecture
plot_model(model, to_file='pima_nn.png', show_shapes=True, show_layer_names=True)
