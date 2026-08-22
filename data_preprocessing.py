import pandas as pd
import seaborn as sns 
df=sns.load_dataset('titanic')
print(df.shape)
print(df.head())
print(df.isnull().sum())
#------------------------Handling Missing Values---------------------
#Technique 1:Droping rows/columns with missing data
df_dropped_rows=df.dropna()
df_dropped_col=df.drop(columns=['deck'])
#print(" \n",df.isnull().sum())

#Technique 2:Mean/Median Imputation
from sklearn.impute import SimpleImputer
#Age had 177 missing Values
#imputer_mean=SimpleImputer(strategy='mean')
#df['age']=imputer_mean.fit_transform(df[['age']])
#print("\n",df['age'])
#print("\n",df.isnull().sum()) now 0 missing values
#Median is Often Preffered over Mean when Outliers exists
impute_median=SimpleImputer(strategy='median')
df['age']=impute_median.fit_transform(df[['age']])
#print(df['age'].head(20))
#print(df.isnull().sum())

#Technique 3:Mode(most common value) Imputation (Categorical Columns)
imputer_mode=SimpleImputer(strategy='most_frequent')
df['embarked']=imputer_mode.fit_transform(df[['embarked']]).ravel()
#print(df.isnull().sum())

#Technique 4:Constant/PlaceHolder fill
#df['deck']=df['deck'].fillna('Unknown')
#Will replace Nan with the unknown in the dataset
#print(df.isnull().sum())

#Technique 5:KNN Imputation(smarter,uses relationships between features)
from sklearn.impute import KNNImputer
knn_imputer=KNNImputer(n_neighbors=5)
df[['age','fare']]=knn_imputer.fit_transform(df[['age','fare']])
#Instead of using one global average, 
# KNN imputation looks at the 5 most similar passengers 
# (by other features) and fills in the missing value based 
# on their values — a more context-aware guess 
# than a blanket average.

#--------------------------Feature Scaling--------------------------
# i) Z-Score Normalization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['age','fare']]=scaler.fit_transform(df[['age','fare']]) 
# ii) MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
df[['age','fare']]=minmax.fit_transform(df[['age','fare']])
# CORRECT approach:
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
#X_test_scaled = scaler.transform(X_test)          # ONLY transform on test (reuse train's mean/std)

#--------------------------------Encoding--------------------------------------
# i) Label Encoding(for ordinal/binary categories)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['sex_encoded']=le.fit_transform(df['sex'])

# ii) One Hot Encoding(for nomial/unordered categories)
df_encoded=pd.get_dummies(df,columns=['embarked'],drop_first=True)
 #-------------------Train/Validation/Test Split-------------------
from sklearn.model_selection import train_test_split
X=df.drop(columns=['survived'])
y=df['survived']
#First Split:(seperate out test set (e.g 20%))
X_temp,X_test,y_temp,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Second Split:split remaining 80% into train/validation
X_train,X_val,y_train,y_val=train_test_split(X_temp,y_temp,test_size=0.25,random_state=42)
