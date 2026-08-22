import seaborn as sns 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
df=sns.load_dataset('titanic')
df=df[['survived','pclass','sex','age','fare','embarked']]
df['age']=SimpleImputer(strategy='median').fit_transform(df[['age']])
df['embarked']=SimpleImputer(strategy='most_frequent').fit_transform(df[['embarked']]).ravel()
df=pd.get_dummies(df,columns=['sex','embarked'],drop_first=True)
X=df.drop(columns=['survived'])
y=df['survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
print("Ready for training:", X_train_scaled.shape, X_test_scaled.shape)