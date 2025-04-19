## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by : Prabhakaran P
Reg No : 212224040236
```
```
from google.colab import drive
drive.mount('/content/drive')
ls drive/MyDrive/'Encoding Data.csv'
```
```
import pandas as pd
df=pd.read_csv('drive/MyDrive/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/fe3d223d-4185-4803-a9de-33f4176a5a48)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/1aaa8cb3-bf87-4b3e-879d-51a133b36634)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/118954e1-3833-44aa-91e1-adec3e6bf698)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/972b7dbe-3dd5-4ffb-8071-0e327a602a9d)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
enc
```
![image](https://github.com/user-attachments/assets/93cc721a-b95a-4829-800e-311f2587d6e2)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/bb88ada4-5691-4d6f-af34-280e8ac02883)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/c4f3eb4f-9888-4bed-ab5d-f9a3018c86a1)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/20bd77fb-9f20-439f-8cbe-ee83e6d97f78)

```
ls drive/MyDrive/data.csv
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv('drive/MyDrive/data.csv')
df
```
![image](https://github.com/user-attachments/assets/eb03e424-d47a-4103-bde1-f7de6bda4f3a)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/5e315caa-5313-44ff-9574-6012711fbb10)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/078a3994-f400-42a3-853d-8facc133270d)
```
ls drive/MyDrive/Data_to_Transform.csv
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('drive/MyDrive/Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/7d41ad2f-688c-487d-9de6-e6c31f8d2241)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/af36b3df-736b-4134-ae38-5589aa2d5946)
```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/4922dabc-0558-459b-bcba-11a0df31ddf4)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/9992e44a-48ff-423b-b07f-a510a6f08a77)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/aecc0e5b-01f6-4fe8-965c-f874a8f5deb6)
```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/5118c9d5-8c75-4989-b83f-3a1a81c092d3)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/cd8afe27-3b14-4fd6-add9-61270f96ec65)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/eca40ecf-e98c-42a2-aec4-ea0e6822152d)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/64d17732-df64-421f-a6c3-d673bd27570e)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/972ce003-7bd5-4a18-835c-e67d46d53a24)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/e1350099-8c10-4589-984f-efc65abe8eb2)
```
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


![download](https://github.com/user-attachments/assets/00a80903-83ff-4923-b2cb-141d75454058)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/fc22c8eb-3ded-41ab-ba9b-09ec46f528e6)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/f12ed8ad-b23f-4065-ac10-2c4988a9e14f)
```
```

# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
```
       
