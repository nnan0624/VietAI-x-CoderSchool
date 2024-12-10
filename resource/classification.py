import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("report.html")
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(random_state=100)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))


