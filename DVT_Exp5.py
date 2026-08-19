#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Churn_dataset.csv")

print("Dataset Loaded Successfully!")
df.head()


# In[3]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# In[4]:


sample = df.head(20)

plt.figure(figsize=(8,5))

plt.plot(sample["CustomerId"], sample["Balance"], marker="o")

plt.title("Customer Balance Trend")
plt.xlabel("Customer ID")
plt.ylabel("Balance")

plt.show()


# In[5]:


geography_count = df["Geography"].value_counts()

plt.figure(figsize=(8,5))

plt.bar(geography_count.index, geography_count.values)

plt.title("Customers by Geography")
plt.xlabel("Geography")
plt.ylabel("Number of Customers")

plt.show()


# In[6]:


churn_count = df["Exited"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    churn_count.values,
    labels=["Stayed", "Exited"],
    autopct="%1.1f%%"
)

plt.title("Customer Churn Distribution")

plt.show()


# In[7]:


plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=15)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()


# In[8]:


plt.figure(figsize=(8,5))

plt.scatter(df["Age"], df["Balance"])

plt.title("Age vs Balance")
plt.xlabel("Age")
plt.ylabel("Balance")

plt.show()


# In[9]:


plt.figure(figsize=(8,5))

plt.boxplot(df["Balance"])

plt.title("Balance Distribution and Outliers")
plt.ylabel("Balance")

plt.show()


# In[10]:


sample = df.head(10)

plt.figure(figsize=(8,5))

plt.plot(
    sample["CustomerId"],
    sample["Balance"],
    marker="o",
    label="Balance"
)

plt.plot(
    sample["CustomerId"],
    sample["EstimatedSalary"],
    marker="o",
    label="Estimated Salary"
)

plt.title("Balance vs Estimated Salary")
plt.xlabel("Customer ID")
plt.ylabel("Value")

plt.legend()

plt.show()


# In[ ]:




