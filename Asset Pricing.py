#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:38:19 2023

@author: hiro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame([[0.167, 0.15, 0.142, 0.14, 0.131, 0.12, 0.115, 0.119, 0.118, 0.1010], [0.95, 0.89, 0.83, 0.84, 0.88, 0.88, 0.99, 1, 1.05, 1.1], [1,2,3,4,5,6,7,8,9,10]])

df2 = df.transpose()


df2.columns = ["avg", "beta", "decile"]
df2["risk_premium"] = df2["avg"] - 0.058

df2["decile"] = df2["decile"].astype('category')
import seaborn as sns

sns.set_theme()

sns.relplot(
    data=df2,
    x="beta", y="risk_premium", size = "decile", hue = "decile"
)
sns.lmplot(x="beta", y="risk_premium", data=df2)

sns.scatterplot(
    data=df2,
    x="beta", y="risk_premium", size = "decile", hue = "decile"
)

plt.xlabel(r'$\beta$')
plt.ylabel('Average Risk Premium')
plt.title('Security Market Line')

df2["beta_difference"] = [0.025, -0.12, -0.14, -0.19, -0.4, -0.56, -0.83, -0.78, -0.88, -1.23]




from sklearn.linear_model import LinearRegression

x = df2[["beta", "beta_difference"]]
y = df2["risk_premium"]
reg = LinearRegression().fit(df2[["beta", "beta_difference"]], df2["risk_premium"])

r_sq = reg.score(x, y)
print(f"coefficient of determination: {r_sq}")
#coefficient of determination: 0.9998492770736408

print(f"intercept: {reg.intercept_}")
#intercept: -0.001425767067272335

print(f"coefficients: {reg.coef_}")
coefficients: [0.1142437  0.06616563]




