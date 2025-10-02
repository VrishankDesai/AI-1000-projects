#Implement logistic regression model to classify a dataset(ex-predicting if customer will make a purchase)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

#Generate synthetic dataset
np.random.seed(42)
n_samples = 200
X=np.random.rand(n_samples,2)*10
y=(X[:,0]*1.5 + X[:,1] > 15).astype(int)

#Create a dataframe
df = pd.DataFrame(X,columns=['Age','Salary'])
df['Purchase']=y

#Split data
X_train,X_test,y_train,y_test=train_test_split(df[['Age','Salary']],df['Purchase'],test_size=0.2,random_state=42)

#Train logistic model
model = LogisticRegression()
model.fit(X_train,y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate performance
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Precision: ",precision_score(y_test,y_pred))
print("Recall: ",recall_score(y_test,y_pred))
print("F1 Score: ",f1_score(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))

#Plot decision boundary
x_min,x_max = X['Age'].min() - 1, X['Age'].max() + 1
y_min,y_max = X['Salary'].min() - 1, X['Salary'].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max,0.1))

#Predict probabilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#Plot
plt.contour(xx,yy,Z,alpha=0.8,cmap="coolwarm")
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolors="k", cmap="coolwarm")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()