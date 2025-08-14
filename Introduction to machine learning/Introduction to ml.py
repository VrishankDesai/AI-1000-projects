import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

#define features and target
features=df[['total_bill','size']]
target=df['tip']

print("Features: \n",features.head(20))
print("Target: \n",target.head(20))

x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=0)

print("Training dataset: ",x_train.shape)
print("Testing dataset: ",x_test.shape)

#Visualize Relationships
sns.pairplot(df,x_vars=["total_bill","size"],y_vars="tip",height=5,aspect=0.8,kind="scatter")
plt.title("Feature vs Target Relationships")
plt.show()