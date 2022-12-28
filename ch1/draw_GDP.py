
import os
datapath = os.path.join("..", "book", "datasets", "lifesat", "")

import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=",", delimiter='\t', encoding='latin1', na_values="n/a")


contry_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[contry_stats["GDP per capita"]]
y = np.c_[contry_stats["Life satisfaction"]]
print(x)


contry_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
#plt.show()


model = sklearn.linear_model.LinearRegression()

model.fit(x, y)

t0, t1 = model.intercept_[0], model.coef_[0][0]
X = np.linspace(0, 60000, 1000)
plt.plot(X, t0+ t1*X, "b")
plt.show()

x_new = [[22587]]
print(model.predict(x_new))
