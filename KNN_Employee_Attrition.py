# KNN Classification
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler

trainCSVFile = 'https://raw.githubusercontent.com/dambi-ai/GEN-AI/refs/heads/Bronze/employee_attrition_train.csv'
df = read_csv(trainCSVFile)
#print(df)

if df['JobRole'].dtype == 'object':
  le = LabelEncoder()
  df['JobRole'] = le.fit_transform(df['JobRole'])

input = df.iloc[:,0:5].values
output = df.iloc[:,5].values
print("input \n",input)
print("output \n",output)

thismodel = KNeighborsClassifier()
print("\nThe parameters of the model are\n\n",thismodel.get_params())
thismodel.fit(input,output)


testCSVFile = 'https://raw.githubusercontent.com/dambi-ai/GEN-AI/refs/heads/Bronze/employee_attrition_test.csv'
newdataframe = read_csv(testCSVFile)

if newdataframe['JobRole'].dtype == 'object':
   newdataframe['JobRole'] = le.fit_transform(newdataframe['JobRole'])

testinputz = newdataframe.iloc[0:4,0:5].values

print("\n csv test file \n",newdataframe)
print("\n test input \n",testinputz)



predResult=thismodel.predict(testinputz)
predList=[]
for val in predResult:
    if val==0:
        predList.append("Attrition_No")
    else:
        predList.append("Attrition_Yes")

print("\n Prediction List ",predList)

print("\n Predicition Result ",predResult)
