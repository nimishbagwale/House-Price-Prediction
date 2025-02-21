import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
import datetime
from sklearn.tree import DecisionTreeRegressor

current_year = datetime.datetime.now().year

dataset = pd.read_csv("data.csv")
dataset = dataset[dataset['price'] > 0]

dataset["house_age"] = current_year - dataset["yr_built"]
dataset["renovated_age"] = dataset.apply(lambda x: current_year - x["yr_renovated"] if x["yr_renovated"] > 0 else 0 , axis=1)
dataset["was_renovated"] = dataset.apply(lambda x: 1 if x["renovated_age"] > 0 else 0 , axis=1)
dataset["log_sqft_living"] = np.log1p(dataset["sqft_living"])
dataset["log_sqft_lot"] = np.log1p(dataset["sqft_lot"])
dataset["log_sqft_above"] = np.log1p(dataset["sqft_above"])
city_avg_price = dataset.groupby("city")["price"].mean()
dataset["city_avg_price"] = dataset["city"].map(city_avg_price)



dataset = dataset.drop(["yr_built","yr_renovated","date","waterfront","street","country","statezip"],axis=1)

y = dataset.iloc[:,0].values
x = dataset.drop(["price"],axis=1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), ["city"])], remainder='passthrough')
x = pd.DataFrame(ct.fit_transform(x))

# print(y.shape)
# print(x[46].shape)
# plt.scatter(x.loc[46],y[0],color="green")
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# top R2 score - 0.6535735388360642
poly = PolynomialFeatures(degree=2, interaction_only=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)
y_pred = regressor.predict(x_test_poly)
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(result)
print(f"Your R2 Score : {r2_score(y_test,y_pred)}")


# print(x_train)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1)).ravel()
# y_test = sc_y.fit_transform(y_test.reshape(-1,1)).ravel()
# regressor = SVR(kernel='rbf')
# regressor.fit(x_train,y_train)
# y_pred = regressor.predict(x_test)
# result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(result)
# print(f"Your R2 Score : {r2_score(y_test,y_pred)}")

# top R2 score - 0.525
# regressor = LinearRegression()
# regressor.fit(x_train,y_train)
# y_pred = regressor.predict(x_test)
# result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(result)
# print(f"Your R2 Score : {r2_score(y_test,y_pred)}")

# top R2 score - 0.41
# regressor = DecisionTreeRegressor(random_state=30)
# regressor.fit(x_train,y_train)
# y_pred = regressor.predict(x_test)
# result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(result)
# print(f"Your R2 Score : {r2_score(y_test,y_pred)}")



# top R2 score - 0.45
# rf_regressor = RandomForestRegressor(n_estimators=200, random_state=0)
# rf_regressor.fit(x_train, y_train)
# y_pred_rf = rf_regressor.predict(x_test)
# print(f"Random Forest R2 Score: {r2_score(y_test, y_pred_rf)}")


