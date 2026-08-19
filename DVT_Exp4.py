#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("ChurnModelling.csv")

print("Dataset Loaded Successfully!")
df.head()


# In[3]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# In[4]:


print(df.dtypes)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


high_balance = df[df["Balance"] > 100000]

print(high_balance.head())


# In[8]:


high_credit = df[df["CreditScore"] > 700]

print(high_credit.head())


# In[9]:


sorted_df = df.sort_values(
    by="Balance",
    ascending=False
)

sorted_df[["CustomerId", "Balance"]].head(10)


# In[10]:


print("Mean Balance =", df["Balance"].mean())
print("Median Balance =", df["Balance"].median())
print("Mode Balance =", df["Balance"].mode().iloc[0])
print("Standard Deviation =", df["Balance"].std())


# In[11]:


print(df[["CreditScore", "Age", "Balance", "EstimatedSalary"]].describe())


# In[12]:


geography_avg = df.groupby("Geography")["Balance"].mean()

print(geography_avg)


# In[13]:


geography_avg.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average Balance by Geography")
plt.xlabel("Geography")
plt.ylabel("Average Balance")

plt.show()


# In[14]:


corr = df[
    ["CreditScore", "Age", "Tenure", "Balance",
     "NumOfProducts", "EstimatedSalary", "Exited"]
].corr()

print(corr)


# In[15]:


print(corr["Exited"].sort_values(ascending=False))


# In[16]:


plt.figure(figsize=(9, 6))

plt.imshow(corr, cmap="coolwarm", aspect="auto")

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Matrix")

plt.show()


# In[17]:


plt.figure(figsize=(8, 5))

plt.scatter(df["Age"], df["Balance"])

plt.title("Age vs Balance")
plt.xlabel("Age")
plt.ylabel("Balance")

plt.show()


# In[ ]:




