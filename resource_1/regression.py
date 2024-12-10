import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")
target = "math score"

x = data.drop(target, axis=1)
y = data[target]

# print(data[["math score", "reading score", "writing score"]].corr())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

education_order = ["some high school", "high school", "some college",
                   "associate's degree", "bachelor's degree", "master's degree"]

gender_order = x_train["gender"].unique()
lunch_order = x_train["lunch"].unique()
test_order = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_order, gender_order, lunch_order, test_order]))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"])
])

model = Pipeline(steps=[
    ("pre_processor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=100))
])

# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
#
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))

# params = {
#     "regressor__n_estimators": [50, 100, 200],
#     "regressor__criterion": ["squared_error", "absolute_error", "friedman_mse"],
#     # "regressor__max_depth": [None, 2, 5, 10],
#     "pre_processor__num_features__imputer__strategy": ["mean", "median"]
# }
#
# model_gr = GridSearchCV(model, param_grid=params,
#                         scoring="r2", cv=6, verbose=1, n_jobs=4)
#
# model_gr.fit(x_train, y_train)
# print("Best score: {}".format(model_gr.best_score_))
# print("Best params: {}".format(model_gr.best_params_))

# Train and test multiple models at the same time
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
models, predictions = reg.fit(x_train, x_test, y_train, y_test)


