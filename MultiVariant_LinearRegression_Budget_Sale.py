import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('https://raw.githubusercontent.com/dambi-ai/GEN-AI/refs/heads/Bronze/advertise_budget_train.csv')

print("print data is \n",dataset)

inputx = dataset.iloc[:, 0:3].values
outputy = dataset.iloc[:, 3].values
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 7)

print("\n print input is \n",inputx)
print("\n print output is \n",outputy)

model = LinearRegression()
print("\nThe parameters of the model are\n\n",model.get_params())

print("\nThe model we are using is ", model.fit(input_train, output_train))

print("\n input_test \n",input_test)

y_pred = model.predict(input_test)

print("\n y_pred",y_pred)

example = np.array([[200,30,50]])
predicate_sale = model.predict(example)

print("\n Predicted sale for TV=200,Radio=30,Newspaper=50 : ",predicate_sale[0])
