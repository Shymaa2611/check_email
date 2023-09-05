"""
Created on Tue Sep  5 17:30:02 2023

@author: shymaa
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
data=pd.read_csv('D:\\MachineCourse\\MachineLearnig\\NLP\\NLP-main\\Code Section 05\\smsspamcollection.tsv',sep='\t')

print(data.head())
#======================= number of ham and spam meassage ======================#
print(data['label'].value_counts())
print(data.info())
print(data.describe())
print(data.isnull())
print(data.isnull().sum())

x=data['message']
y=data['label']
#============================ x values ==================================#
print(x.head())
#======================= y values =======================================#
print(y.head())

x_train,x_test,y_train,y_test= train_test_split(x, y, test_size=0.25, random_state=42)
#==== test = 25 % from data and train 75% from data#
#===================== classification meassage into spam and ham ================#
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])
text_clf.fit(x_train, y_train)  
y_predict = text_clf.predict(x_test)
print(y_predict)
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))

#print(metrics.mean_absolute_error(y_test, y_predict)) 

cm=metrics.confusion_matrix(y_test,y_predict)
sns.heatmap(cm,center=True)
#================== calucate accuracy of algorithm =================#
print("accuracy = ",metrics.accuracy_score(y_test, y_predict))
print(text_clf.predict(['Hi How are you ? ']))
print(text_clf.predict([' hello ahmed how are you now ? ']))
print(text_clf.predict([' Congratulations , you won the dream prize']))




