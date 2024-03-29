#!/usr/bin/env python
# coding: utf-8

# In[62]:


#Importing the import Python Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 1: PREPROCESSING

# In[63]:


df=pd.read_csv('creditcard_2023.csv') 


# In[64]:


df.head() #checking if the dataset is loaded properly


# In[65]:


df.tail() # tail function returns the last last five records


# In[66]:


df.shape #checking the total no of records and columns of the given data frame


# In[67]:


df.info() #This gives the information regarding  the null values count in each column as well as the datatype in each column


# The total no of rows are 568630 and the non-null records count are also the same for each column that means the dataset doesn'y consists of any null values

# Another method by which null values can be identified is given as following:

# In[68]:


# Checking for the null values
df.isnull().sum()


# The above output shows that there are no null values so no replacement required

# In[69]:


#checking the duplicate records:
df[df.duplicated()]


# The above output shows that there are no duplicated records 

# In[70]:


# The data in the class column seems to be categorical , checking for the different labelled classes it is distributed in using value_counts
#It basically gives us the count of the unique values in a column
df['Class'].value_counts()


# From the output we can conclude that there are only 2 labelled groups ( non-fraudulent and fraudulent) and boyh have equal counts this means that the data is balanced

# In[71]:


#Describe gives us the mean,standard deviation,count,maximum and minimum values of the features
df.describe()


# In[72]:


#Dropping irrelevant columns :
#Let us 1st copy the dataframe into another one as the authentic data might be required later on:
df_1=df.drop('id',axis=1)


# In[73]:



#Id is irrelavant column as our problem statement states that we are supposed to find the features on the basis of which the credit card benificiary would be predicted as fraudulent or non-fraudulent and id no has nothing to with it.
df_1


#  # Step 2: EDA

# In[74]:


#heat map
corr=df_1.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True)


# The correlation plot shows that the the dependancies of the features V3,V4,V10,V11,V12,V13,V14 is considerable as it is around 0.7 and above. So these features can be chosen for prediction.

# In[75]:


sns.countplot(x='Class',data=df)
plt.xlabel("Fradulent or not")
plt.ylabel("count")
plt.title("Fradulent vs Non-Fradulent")


# In[76]:


# Plotting distribution of transaction amount by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False)
plt.title('Distribution of Transaction Amount by Class')
plt.show()


# # Step 3: MODELLING

# 3.1:Feature Selection on the basis of Graphical Analysis

# MODEL 1:LOGISTIC REGRESSION

# In[77]:


#Using all the features:
x=df_1.iloc[:,:-1]
y=df_1.iloc[:,-1]


# In[78]:


x


# In[79]:


y


# In[80]:


#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[81]:


from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1.fit(x_train,y_train)


# In[82]:


y_pred=l1.predict(x_test)


# In[83]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,y_pred))


# In[84]:


print(classification_report(y_test,y_pred))


# In[85]:


print(accuracy_score(y_test,y_pred)) #testing accuracy


# In[86]:


y_pred_train=l1.predict(x_train)


# In[87]:


print(accuracy_score(y_train,y_pred_train)) #training accuracy


# In[88]:


#using Selected features
x=df_1.iloc[:,[2,3,9,10,11,12,13]]
y=df_1.iloc[:,-1]


# In[89]:


x


# In[90]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[91]:


from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1.fit(x_train,y_train)


# In[92]:


y_pred=l1.predict(x_test)


# In[93]:


accuracy_score(y_pred,y_test)


# In[94]:


print(confusion_matrix(y_test,y_pred))


# In[95]:


print(classification_report(y_test,y_pred))


# In[96]:


#Decision Trees
from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)


# In[97]:


y_pred_dtc=dtc.predict(x_test)


# In[98]:


accuracy_score(y_pred,y_test)


# In[99]:


print(confusion_matrix(y_test,y_pred))


# In[100]:


print(classification_report(y_test,y_pred))


# In[101]:


from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier(criterion='entropy',random_state=10)
dtc.fit(x_train,y_train)


# In[102]:


y_pred_dtc=dtc.predict(x_test)


# In[103]:


accuracy_score(y_pred,y_test)


# In[104]:


print(confusion_matrix(y_test,y_pred))


# In[105]:


print(classification_report(y_test,y_pred))


# In[106]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier


# Assuming you have a DataFrame 'X' containing features and a Series 'y' containing labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=0)

# Train the model
rf_classifier.fit(x_train, y_train)


# In[107]:


y_pred_tree=rf_classifier.predict(x_test)
y_pred_tt=rf_classifier.predict(x_train)


# In[108]:


accuracy_score(y_pred_tt,y_train)


# In[109]:


accuracy_score(y_pred_tree,y_test)


# In[110]:


print(confusion_matrix(y_test,y_pred_tree))


# In[111]:


print(classification_report(y_test,y_pred_tree))


# In[112]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier


# Assuming you have a DataFrame 'X' containing features and a Series 'y' containing labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=150, random_state=0)

# Train the model
rf_classifier.fit(x_train, y_train)


# In[113]:


y_pred_tree=rf_classifier.predict(x_test)
y_pred_tt=rf_classifier.predict(x_train)


# In[114]:


accuracy_score(y_pred_tt,y_train)


# In[115]:


print(confusion_matrix(y_test,y_pred_tree))


# In[116]:


print(classification_report(y_test,y_pred_tree))


# In[117]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Create an SVC model with a linear kernel
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf',random_state=0)
svm_model.fit(x_train, y_train)


# Train the model
svm_model.fit(x_train, y_train)

# Make predictions
predictions = svm_model.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[118]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[119]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[120]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[121]:


#ANN 
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=32, epochs=100)
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)


# In[123]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[124]:


# Display additional metrics
print(classification_report(y_test, y_pred))


# In[125]:


#ANN 
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=32, epochs=50)
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)


# In[126]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[127]:


# Display additional metrics
print(classification_report(y_test, y_pred))


# In[128]:


#ANN 
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=25, epochs=50)
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)


# In[129]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[130]:


# Display additional metrics
print(classification_report(y_test, y_pred))


# In[131]:


#PCA
X = df_1.iloc[:,:-1]
#Step 1: Standardization
X_mean = X.mean()
X_std = X.std()
z = (X - X_mean)/X_std


# In[132]:


#Step 2: Covariance matrix
corr = z.cov()

import seaborn as sns
sns.heatmap(corr)


# In[133]:


#Step 3: Eigen values and Eigen vector

eigenvalues, eigenvectors = np.linalg.eig(corr)


# In[134]:


#Indexing
idx = eigenvalues.argsort()[::-1]


# In[135]:


#Sorting of eigen values
eigenvalues = eigenvalues[idx]


# In[136]:


#Sorting of eigen vectors
eigenvectors = eigenvectors[:,idx]


# In[137]:


variance = np.cumsum(eigenvalues)/np.sum(eigenvalues)
variance


# In[138]:


#Step 4 : Number of components
n_component = np.argmax(variance >=0.50) + 1
n_component


# In[139]:


#Apply PCA
from sklearn.decomposition import PCA
dr = PCA(n_components=3)
dr.fit(z)
dr_pca = dr.transform(z)
df1 = pd.DataFrame(dr_pca, columns=['PC{}'.
                                format(i+1) 
                                for i in range (n_component)])


# In[140]:


df1


# In[141]:


#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df1,y,test_size=0.3,random_state=0)


# In[142]:


#Logistic
from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
l1.fit(x_train,y_train)
y_pred=l1.predict(x_test)


# In[143]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,y_pred))


# In[144]:


print(classification_report(y_test,y_pred))


# In[145]:


accuracy_score(y_pred,y_test)


# In[147]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=150, random_state=0)

# Train the model
rf_classifier.fit(x_train, y_train)
#Prediction:
y_pred=rf_classifier.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_pred,y_test)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=0)

# Train the model
rf_classifier.fit(x_train, y_train)
y_pred=rf_classifier.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_pred,y_test)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_pred,y_test)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_pred,y_test)


# In[ ]:


# SVC
svm_model = SVC(kernel='rbf',random_state=0)
svm_model.fit(x_train, y_train)


# Train the model
svm_model.fit(x_train, y_train)

# Make predictions
predictions = svm_model.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, random_state=0)

# Train the model
gb_classifier.fit(x_train, y_train)

# Make predictions
predictions = gb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics
print(classification_report(y_test, predictions))


# In[ ]:


#ANN
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=32, epochs=100)
y_pred = ann.predict(x_test) 
y_pred = (y_pred > 0.5)


# In[ ]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# In[ ]:



# Display additional metrics
print(classification_report(y_test, predictions))


# In[ ]:


#ANN
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=32, epochs=50)
y_pred = ann.predict(x_test) 
y_pred = (y_pred > 0.5)


# In[ ]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[ ]:


# Display additional metrics
print(classification_report(y_test, y_pred))


# In[ ]:


#ANN
import tensorflow as tf
import keras
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding Input layer 
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#Adding second hidden line
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size=25, epochs=50)
y_pred = ann.predict(x_test) 
y_pred = (y_pred > 0.5)


# In[ ]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[ ]:


# Display additional metrics
print(classification_report(y_test, y_pred))


# In[ ]:


#KNN : without cross validation
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)


# In[ ]:


y_pred = knn.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


#Cross Validation
#Choosing a k value
from sklearn.model_selection import cross_val_score
accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,df1,y,cv=10)
    accuracy_rate.append(score.mean())


# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,df1,y,cv=10)
    error_rate.append(1-score.mean())


# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i !=y_test))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy_score(y_pred,y_test)


# #Conclusion: 
# ANN algorithm is better than any other algorithm as the sum of products of all the inputs and their weights are calculated, which is later fed to the output. So the bias and the multicollinearity factor between the numerous features are easily taken care off.
# The transformation of the features using PCA yeilds us the best result as there are numerous anonymous features so while claiming on one particular feature dependency on traget seems unreasonable since we can't name it, so it's better we reduce the components.
