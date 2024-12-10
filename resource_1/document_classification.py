import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def filter_location(location):
    result = location.split(",")
    if len(result) > 1:
        return result[1][1:]
    else:
        return location


data = pd.read_excel("job_dataset.ods", engine="odf", dtype="str")
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
# processed_data = vectorizer.fit_transform(x_train["description"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(processed_data.shape)
#uni gram: 66674
# encoder = OneHotEncoder()
# processed_data = encoder.fit_transform(x_train[["industry"]])
# print(processed_data.shape)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "description"),
    ("function", OneHotEncoder(), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

model = Pipeline(steps=[
    ("pre_processor", preprocessor),

    ("regressor", RandomForestClassifier(random_state=100))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))


