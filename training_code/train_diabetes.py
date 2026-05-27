#load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#import all the required library for machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

#load the dataset
data = pd.read_csv("dataset/diabetes.csv")

#print the 5 rows
data.head()

#check the dataset shape
data.shape

#check the dataset datatype
data.info()

#check the null value in the dataset
data.isnull().sum()

#check the describe of datafame
data.describe()

#Drop the duplicates values
data = data.drop_duplicates()

#check the data is balanced or not
sns.countplot(x = "Outcome", data = data)

data.Outcome.value_counts()

#Check the distribution are normal or skewed 
fig,axs = plt.subplots(4,2, figsize = (20,50))

sns.distplot(data.Pregnancies, ax=axs[0,0], color="green")
sns.distplot(data.Glucose, ax=axs[0,1], color="red")
sns.distplot(data.BloodPressure, ax=axs[1,0], color="yellow")
sns.distplot(data.SkinThickness, ax=axs[1,1], color="violet")
sns.distplot(data.Insulin, ax=axs[2,0],color="lime")
sns.distplot(data.BMI, ax=axs[2,1],color="olive")
sns.distplot(data.DiabetesPedigreeFunction, ax=axs[3,0],color="purple")
sns.distplot(data.Age, ax=axs[3,1], color="orange")

#Correlation between data
sns.heatmap(data.corr(), annot=True)

#Check the missing value of data
fig, axs = plt.subplots(4,2, figsize = (15,25))

sns.boxplot(x="Pregnancies", data = data, ax=axs[0,0])
sns.boxplot(x="Glucose", data = data, ax = axs[0,1])
sns.boxplot(x="BloodPressure", data = data,ax=axs[1,0])
sns.boxplot(x="SkinThickness", data = data, ax=axs[1,1])
sns.boxplot(x="Insulin", data = data, ax= axs[2,0])
sns.boxplot(x="BMI", data = data,ax=axs[2,1])
sns.boxplot(x="DiabetesPedigreeFunction", data = data, ax= axs[3,0])
sns.boxplot(x="Age", data = data, ax=axs[3,1])

#scale down the data
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
new_data = scl.fit_transform(data)

new_data = pd.DataFrame(new_data)
new_data

#removing the outliers
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer()
df = qt.fit_transform(new_data)
df = pd.DataFrame(df)
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df

#check the outliers are removed or not
fig, axs = plt.subplots(4,2, figsize = (15,25))

sns.boxplot(x="Pregnancies", data = df, ax=axs[0,0])
sns.boxplot(x="Glucose", data = df, ax = axs[0,1])
sns.boxplot(x="BloodPressure", data = df,ax=axs[1,0])
sns.boxplot(x="SkinThickness", data = df, ax=axs[1,1])
sns.boxplot(x="Insulin", data = df, ax= axs[2,0])
sns.boxplot(x="BMI", data = df,ax=axs[2,1])
sns.boxplot(x="DiabetesPedigreeFunction", data = df, ax= axs[3,0])
sns.boxplot(x="Age", data = df, ax=axs[3,1])

#histogram plot
fig,axs = plt.subplots(4,2, figsize = (20,50))

sns.distplot(df.Pregnancies, ax=axs[0,0], color="green")
sns.distplot(df.Glucose, ax=axs[0,1], color="red")
sns.distplot(df.BloodPressure, ax=axs[1,0], color="yellow")
sns.distplot(df.SkinThickness, ax=axs[1,1], color="violet")
sns.distplot(df.Insulin, ax=axs[2,0],color="lime")
sns.distplot(df.BMI, ax=axs[2,1],color="olive")
sns.distplot(df.DiabetesPedigreeFunction, ax=axs[3,0],color="purple")
sns.distplot(df.Age, ax=axs[3,1], color="orange")

df.Outcome.value_counts()

x = df.drop(columns = "Outcome", axis = 1)
y = df.Outcome

x 

y

#Balance dataset
from imblearn.over_sampling import RandomOverSampler
rs = RandomOverSampler()
x_ref,y_ref = rs.fit_resample(x,y)

print(x_ref.shape,y_ref.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_ref,y_ref, test_size=0.2)

print(x_train.shape,x_test.shape, y_test.shape, y_train.shape)

model_selection = {
    "Logistic Regression" :{
        "model": LogisticRegression(),
        "parameters": {
            "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    },
    'svm' : {
        'model' : svm.SVC(),
        'parameters' : {
            'kernel' : ['rbf','linear'],
            'C' : [10,15,20]
        }
    },
    "Decision Tree" : {
        'model': DecisionTreeClassifier(),
        "parameters" : {
            "criterion" : ['gini','entropy'],
            "max_depth" : [1,5,10,50,100,500,1000,1500],
            "max_leaf_nodes" : [1,5,10,15]
        }
    },
    "Random Forest" : {
        "model" : RandomForestClassifier(),
        "parameters" : {
            "criterion" : ['gini','entropy'],
            "n_estimators" : [50,100,150,200]
        }
    },
    "KNN" : {
        "model" : KNeighborsClassifier(),
        "parameters" : {
            'n_neighbors' : [3,5,7,8,10]
        }
    },
    'naive_bayes_gaussian' : {
        'model' : GaussianNB(),
        'parameters' : {}
    }
}

score = []

for model_name, mp in model_selection.items():
    clf = GridSearchCV(mp['model'], mp['parameters'], cv=10)
    clf.fit(x_train,y_train)
    score.append({
        'model' : model_name,
        'best_score' : clf.best_score_,
        'best_params' : clf.best_params_
    })
    
diab_df = pd.DataFrame(score, columns = ['model', 'best_score', 'best_params'])
diab_df


import joblib

# Save the trained model
joblib.dump(model, "diabetes_model.pkl")

print("Model saved successfully as diabetes_model.pkl")