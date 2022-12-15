import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("train-data.csv")
print(data.head())
print(data.shape[0], data.shape[1])

r_data = data.copy(deep=True)
r_data.dropna('index').shape
print(str(r_data.New_Price.isna().sum()))
r_data.drop(columns = ['Unnamed: 0', 'New_Price'],inplace = True)
r_data.dropna("index", inplace = True)
r_data = r_data.reset_index(drop = True)
print(r_data.shape)

len(np.unique(list(r_data.Name)))
name = list(r_data.Name)
for i in range(len(name)):
    name[i] = name[i].split(' ',1)[0]
r_data.Name = name
print(r_data.head())
print(len(np.unique(list(r_data.Name))))

mil = list(r_data.Mileage)
eng = list(r_data.Engine)
pow = list(r_data.Power)

for i in range(len(name)):
    mil[i] = mil[i].split(' ', 1)[0]
    eng[i] = eng[i].split(' ', 1)[0]
    pow[i] = pow[i].split(' ', 1)[0]

r_data.Mileage = mil
r_data.Engine = eng
r_data.Power = pow
print(r_data.head())

r_data["Price"] = r_data["Price"].astype(float)
r_data["Kilometers_Driven"] = r_data["Kilometers_Driven"].astype(float)
r_data["Mileage"] = r_data["Mileage"].astype(float)
r_data["Engine"] = r_data["Engine"].astype(float)
print(r_data.dtypes)

print(np.unique(list(r_data.Seats)))
r_data = r_data[r_data.Seats != 0]
print(np.unique(list(r_data.Seats)))
print(r_data.shape)

print(np.unique(list(r_data.Power)))

idx = []
lt = list(r_data['Power'])
for i in range(len(lt)):
    if (lt[i] == 'null'):
        idx.append(i)

r_data = r_data.drop(columns = 'Power', axis = 1)

#r_data = r_data.drop(idx)
#r_data = r_data.reset_index(drop = True)

print(r_data.isnull().sum())

r_data['Year'] = pd.Categorical(r_data['Year'])
r_data['Seats'] = pd.Categorical(r_data['Seats'])
r_data = pd.get_dummies(r_data, prefix_sep='_', drop_first = True)
print(r_data.head())

print(r_data.shape)

idx = []
lt = list(r_data['Kilometers_Driven'])
for i in range(len(lt)):
    if (lt[i] > 1000000):
        idx.append(i)
r_data = r_data.drop(idx)
r_data = r_data.reset_index(drop = True)

fig, ax = plt.subplots(1,4,figsize=(16,3))
ax[0].boxplot(list(r_data.Kilometers_Driven))
ax[0].set_title("Kilometers Driven")

ax[1].boxplot(r_data.Mileage)
ax[1].set_title("Mileage")

ax[2].boxplot(r_data.Engine)
ax[2].set_title("Engine")

ax[3].boxplot(list(r_data.Price))
ax[3].set_title("Price")

plt.show()

sns.pairplot(data=r_data, x_vars=['Kilometers_Driven', 'Mileage', 'Engine'], y_vars='Price', size=3)
plt.show()

y = r_data[['Price']].to_numpy()
r_data = r_data.drop(columns= ['Price'])

x = r_data.values
columns = r_data.columns

scaler = MinMaxScaler()
tmp = scaler.fit_transform(x)
r_data = pd.DataFrame(tmp)
r_data.columns = columns

x = r_data.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.85, random_state=1)

lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
lr.fit(x_train, y_train)

print('Train Accuracy: ',format(lr.score(x_train, y_train)))

y_predict = lr.predict(x_test)

print('Test Accuracy: ', format(lr.score(x_test, y_test)))
print('Test Accuracy: ', format(r2_score(y_test,lr.predict(x_test))))

print('mean absolute error: ', mean_absolute_error(y_test, y_predict))