import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only 2 features for visualization
y = iris.target
target_names = iris.target_names

# Streamlit UI
st.title("KNN Classifier on Iris Dataset 🌸")
st.write("Interact with the sliders to classify a flower using KNN!")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
k_value = st.slider("Number of Neighbors (k)", 1, 10, 3)

# Train model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X, y)

# Prediction
sample = np.array([[sepal_length, sepal_width]])
prediction = knn.predict(sample)
predicted_class = target_names[prediction[0]]

st.subheader(f"🌼 Predicted class: **{predicted_class}**")

# Plotting
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)

# Plot training data
colors = ['red', 'green', 'blue']
for i, color in zip(range(3), colors):
    idx = np.where(y == i)
    ax.scatter(X[idx, 0], X[idx, 1], c=color, label=target_names[i], edgecolor='k', s=60)

# Plot input point
ax.scatter(sepal_length, sepal_width, c='black', s=100, edgecolor='white', label='Your Input')
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.legend()
st.pyplot(fig)
