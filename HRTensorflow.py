import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow_addons import losses
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import os

df = pd.read_csv("N:\AI\HR.csv")
df.drop('Over18',axis='columns',inplace=True) #Inplace will update df after this process
df.drop('EmployeeCount',axis='columns',inplace=True) #Inplace will update df after this process
df.drop('EmployeeNumber',axis='columns',inplace=True) #Inplace will update df after this process
df.drop('StandardHours',axis='columns',inplace=True) #Inplace will update df after this process


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}')

print_unique_col_values(df)

yes_no_columns = ["OverTime", "Attrition"]

for col in yes_no_columns:
    df[col].replace({'Yes': 1,'No': 0},inplace=True)

df['Gender'].replace({'Female':1,'Male':0},inplace=True)

df2 = pd.get_dummies(data=df, columns=["BusinessTravel", "Department", "EducationField", "Education", "EnvironmentSatisfaction",
                                        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
                                        "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance"])

cols_to_scale = ['DailyRate','DistanceFromHome','HourlyRate', "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
                 "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

print(df2.dtypes)
print(df2.info())

corrMatrix = df.corr()
f, ax = plt.subplots(figsize =(9, 8))
sn.heatmap(corrMatrix, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
plt.show()
def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(82, input_dim=82, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    if weights == -1:
        model.fit(X_train, y_train, epochs=50)
    else:
        model.fit(X_train, y_train, epochs=50, class_weight=weights)

    print(model.evaluate(X_test, y_test))

    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)

    print("Classification Report: \n", classification_report(y_test, y_preds))
    model.save('N:\AI\my_model.h5')
    return y_preds

# Class count
count_class_0, count_class_1 = df2.Attrition.value_counts()

# Divide by class
df_class_0 = df2[df2['Attrition'] == 0] #Dataframe with all 0 Targets
df_class_1 = df2[df2['Attrition'] == 1] #Dataframe with all 1 Targets

print(df_class_0.shape)
print(df_class_1.shape)

df_class_1_over = df_class_1.sample(count_class_0, replace=True)# This will multiply the dataframe from the 1 Targets. The amount of 1 Targets will be declared the count_class_0 to have equile amount of entries in both dataframes-

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
print(df_test_over.shape)

X = df_test_over.drop('Attrition',axis='columns')
y = df_test_over['Attrition']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_preds)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
