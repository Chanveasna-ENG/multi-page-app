import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_data
def app():
    st.title('Graph')

    st.write('This is the `Graph` page of the multi-page app.')

    st.write('The model performance of the Iris dataset is presented below.')

    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Draw Graph
    st.header('Donut Graph')

    # Draw Pie Chart
    st.header('Pie Chart')
    labels = iris.target_names
    sizes = np.bincount(iris.target)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)

    # Draw Line Chart
    st.header('Line Chart: Showing Average Length of feature for each species')
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = Y
    averages = df.groupby('species').mean()
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(averages.columns, averages.iloc[i], label=iris.target_names[i])
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Length')
    ax.legend()
    st.pyplot(fig)

app()