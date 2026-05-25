import streamlit as st
import pandas as pd
import joblib

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Mall Customer Segmentation")

st.write(
    "Customer segmentation using KMeans Clustering"
)

income = st.slider(
    "Annual Income (k$)",
    0,
    150,
    50
)

spending = st.slider(
    "Spending Score",
    1,
    100,
    50
)

input_data = pd.DataFrame({
    "Annual Income (k$)": [income],
    "Spending Score (1-100)": [spending]
})

scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)[0]

if st.button("Predict Cluster"):

    st.success(
        f"Customer belongs to Cluster {prediction}"
    )