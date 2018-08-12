# TODO: Add import statements
import pandas as p
import numpy as num
from sklearn.linear_model import LinearRegression 

# Assign the dataframe to this variable.
bmi_life_data = p.read_csv('BMI_and_LifeExpectancy.csv')

print(bmi_life_data[['BMI']])
#print(bmi_life_data[['Life expectancy']])
data = num.asarray(bmi_life_data)
#X = data[:,2]
y = data[:,1]

#print(X)
print(y)

bmi_life_model = LinearRegression()
#bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
bmi_life_model.fit(bmi_life_data[['BMI']], y)

# Make a prediction using the model
laos_life_exp = bmi_life_model.predict(21.07931)

print(laos_life_exp)