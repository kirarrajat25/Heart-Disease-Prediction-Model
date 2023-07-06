#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


df = pd.read_csv('cleveland.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

### 1 = male, 0 = female
df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})

df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

import matplotlib.pyplot as plt
import seaborn as sns

# distribution of target vs age 
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

 
# barplot of age vs sex with hue = target
sns.catplot(kind = 'bar', data = df, y = 'age', x = 'sex', hue = 'target')
plt.title('Distribution of age vs sex with the target class')
plt.show()

df['sex'] = df.sex.map({'female': 0, 'male': 1})


################################## data preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[2]:


#########################################   Logistic Regression  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

features = np.array([[52,1,0,125,212,0,1,168,0,1.0,2,2,3]])
prediction = classifier.predict(features)
print("Prediction: {}".format(prediction))

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


# In[3]:


import pandas as pd


# In[4]:


features = pd.DataFrame({
    'age':57,
    'sex':1,
    'cp':4,
    'trestbps':130,
    'chol':131,
    'fbs':0,
    'restecg':0,
    'thalach':115,
    'exang':1,
    'oldpeak':1.2,
    'slope':2,
    'ca':1,
    'thal':7,
    
    
},index=[0])


# In[5]:


features


# In[6]:


p=classifier.predict(features)
if p[0]==0:
    print("NO disease")
else:
    print("disease")


# In[7]:


pip install tkinter


# In[8]:


import joblib


# In[9]:


joblib.dump(classifier,'model_joblib_heart')


# In[10]:


model = joblib.load('model_joblib_heart')


# In[11]:


model.predict(features)


# In[12]:


from tkinter import *
import joblib
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease",font = ('times',18,'bold'), bg = "green", fg = "white").place(x=160,y=500)
        
    else:
        Label(master, text="Possibility of Heart Disease",font = ('times',18,'bold'), bg = "orange", fg = "white").place(x=130,y=500)
        
    
def myreset():
    for widget in master.winfo_children():
        if isinstance(widget, Entry): # If this is an Entry widget class
            widget.delete(0,'end')
            
master = Tk()
master.title("Heart Disease Prediction System")
master.configure(background='black')
master.geometry("500x600")


label = Label(master, text = "    Heart Disease Prediction System"
                      ,font = ('times',20,'bold')    , bg = "black", fg = "white",padx=40,pady=10). \
                               grid(row=0,columnspan=6)
label = Label(master, text = "Enter the following values : "
                  , font = ('times',16,'italic') , bg = "black", fg = "white"). \
                               grid(row=1,column=0)

Label(master, text="Age", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=2,column=0)
Label(master, text="Sex", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=3,column=0)
Label(master, text="Chest Pain", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=4,column=0)
Label(master, text="Trestbps", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=5,column=0)
Label(master, text="Cholestrol", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=6,column=0)
Label(master, text="Fasting Blood Sugar", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=7,column=0)
Label(master, text="Restecg", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=8,column=0)
Label(master, text="Thalach", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=9,column=0)
Label(master, text="Exang", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=10,column=0)
Label(master, text="Oldpeak", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=11,column=0)
Label(master, text="Slope", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=12,column=0)
Label(master, text="CA", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=13,column=0)
Label(master, text="Thalassemia", bg = "black", fg = "white",font = ('times',12,'italic')).grid(row=14,column=0)



e1 = Entry(master,width=20,borderwidth=3)
e2 = Entry(master,width=20,borderwidth=3)
e3 = Entry(master,width=20,borderwidth=3)
e4 = Entry(master,width=20,borderwidth=3)
e5 = Entry(master,width=20,borderwidth=3)
e6 = Entry(master,width=20,borderwidth=3)
e7 = Entry(master,width=20,borderwidth=3)
e8 = Entry(master,width=20,borderwidth=3)
e9 = Entry(master,width=20,borderwidth=3)
e10 = Entry(master,width=20,borderwidth=3)
e11 = Entry(master,width=20,borderwidth=3)
e12 = Entry(master,width=20,borderwidth=3)
e13 = Entry(master,width=20,borderwidth=3)

e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
e3.grid(row=4, column=1)
e4.grid(row=5, column=1)
e5.grid(row=6, column=1)
e6.grid(row=7, column=1)
e7.grid(row=8, column=1)
e8.grid(row=9, column=1)
e9.grid(row=10, column=1)
e10.grid(row=11, column=1)
e11.grid(row=12, column=1)
e12.grid(row=13, column=1)
e13.grid(row=14, column=1)


Button(master, text="Predict",font = ('times',18,'bold'), bg = "red", fg = "white",borderwidth = 10,width=10, command=show_entry_fields).place(x=180,y=420)
Button(master, text="RESET",font = ('times',12,'bold'), bg = "red", fg = "white",borderwidth = 5,width=10, command=lambda:myreset()).place(x=100,y=550)

mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




