import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv('/Users/coderschool/Documents/Projects /VietAI-x-CoderSchool/Week6_7/csgo.csv')

# Preprocessing
X = df.drop(columns=['result'])  # Features
y = df['result']                 # Target

# Encoding categorical target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handling categorical variables (if any in features)
X_encoded = pd.get_dummies(X, drop_first=True)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))





