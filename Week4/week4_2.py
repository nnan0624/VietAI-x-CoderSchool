import pandas as pd
import matplotlib.pyplot as plt

# Example Iris dataset focusing on SepalLength and SepalWidth
iris_sample_data = {
    "SepalLengthCm": [5.1, 4.9, 4.7, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.8],
    "SepalWidthCm": [3.5, 3.0, 3.2, 3.6, 3.2, 3.2, 3.1, 2.3, 2.8, 2.7],
    "Species": [
        "Iris-setosa",
        "Iris-setosa",
        "Iris-setosa",
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-versicolor",
        "Iris-versicolor",
        "Iris-virginica",
        "Iris-virginica",
        "Iris-virginica",
    ],
}

iris_data = pd.DataFrame(iris_sample_data)

# Plot SepalLengthCm vs. SepalWidthCm
plt.figure(figsize=(8, 6))
for species in iris_data["Species"].unique():
    subset = iris_data[iris_data["Species"] == species]
    plt.scatter(
        subset["SepalLengthCm"],
        subset["SepalWidthCm"],
        label=species,
        alpha=0.7,
    )

plt.title("Iris Dataset: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.grid(True)
plt.show()
