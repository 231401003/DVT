#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# In[2]:


df = pd.read_csv("ChurnModelling.csv")

print("Dataset Loaded Successfully!")
df.head()


# In[3]:


print(df.isnull().sum())


# In[4]:


print(df[df.isnull().any(axis=1)])


# In[5]:


numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled successfully!")
print(df.isnull().sum())


# In[6]:


print("Duplicate Rows:", df.duplicated().sum())


# In[7]:


df.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully!")
print("Remaining Rows:", len(df))


# In[8]:


Q1 = df["Balance"].quantile(0.25)
Q3 = df["Balance"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Balance"] < lower) | (df["Balance"] > upper)]

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Limit =", lower)
print("Upper Limit =", upper)

print("\nNumber of Outliers:", len(outliers))
outliers[["CustomerId", "Balance"]].head()


# In[9]:


plt.figure(figsize=(8, 5))

plt.boxplot(df["Balance"])

plt.title("Balance Outlier Detection")
plt.ylabel("Balance")

plt.show()


# In[10]:


scaler = MinMaxScaler()

df["Balance_Normalized"] = scaler.fit_transform(df[["Balance"]])

print(df[["Balance", "Balance_Normalized"]].head())


# In[11]:


encoder = LabelEncoder()

df["Gender_Encoded"] = encoder.fit_transform(df["Gender"])

print(df[["Gender", "Gender_Encoded"]].head())


# In[12]:


mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print("Gender Encoding:")
print(mapping)


# In[13]:


geography_encoded = pd.get_dummies(
    df["Geography"],
    prefix="Geography",
    dtype=int
)

print(geography_encoded.head())


# In[14]:


df = pd.concat([df, geography_encoded], axis=1)

df.head()


# In[15]:


print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# In[16]:


df.head()


# In[17]:


df.to_csv("ChurnModelling_Cleaned.csv", index=False)

print("Cleaned Dataset Saved Successfully!")


# In[ ]:




