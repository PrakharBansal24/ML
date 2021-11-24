#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("train.csv")
data.head()
#%%
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
x=data["tweet"]
y=data["label"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
from sklearn.ensemble import  RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(x_train, y_train)
pri = random_forest.predict(x_test)
print("Random forest\n",accuracy_score(y_test,pri)*100)

# %%
chinchpokli bander, khao gali