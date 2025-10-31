#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

print("=== BASIC INFO ===")
print(df.info())
print("\n=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe(include='all'))

print("\n === MISSING VALUES ===")
print(df.isnull().sum())

numeric = df.select_dtypes(include=[np.number])
print("\n=== CORRELATION WITH SURVIVAL ===")
print(numeric.corr()['Survived'].sort_values(ascending=False))

print("\n=== SURVIVAL BY SEX ===")
print(df.groupby('Sex')['Survived'].mean())

print("\n=== SURVIVAL BY PCLASS ===")
print(df.groupby('Pclass')['Survived'].mean())

print("\n=== SURVIVAL BY AGEGROUP ===")
print(df.groupby('AgeGroup')['Survived'].mean())

df.to_csv("titanic_cleaned_local.csv", index=False)
print("\nSaved cleaned file as titanic_cleaned_local.csv")


# In[ ]:




