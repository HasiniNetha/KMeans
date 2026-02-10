
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Mall Customer Clustering", page_icon="🛍️", layout="wide")
st.title("🛍️ Mall Customer Clustering Prediction")

MODEL_PATH = Path("kmeans_model.pkl")
DATA_PATH = Path("Mall_Customers.csv")
CLUSTERED_PATH = Path("clustered_mall_customers.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_clustered():
    return pd.read_csv(CLUSTERED_PATH)

model = load_model()
df = load_data()
clustered_df = load_clustered()

scaler = StandardScaler()
scaler.fit(df[['Age','Annual Income (k$)','Spending Score (1-100)']])

st.sidebar.header("Customer Input")
age = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
income = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 50)
spending = st.sidebar.slider("Spending Score", 1, 100, 50)

if st.sidebar.button("Predict Cluster"):
    input_df = pd.DataFrame([[age,income,spending]],
        columns=['Age','Annual Income (k$)','Spending Score (1-100)'])
    scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    cluster = int(model.predict(scaled)[0])
    st.success(f"Predicted Cluster: {cluster}")

    fig = px.scatter_3d(
        clustered_df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color=clustered_df['Cluster'].astype(str)
    )
    fig.add_scatter3d(x=[age], y=[income], z=[spending], mode='markers',
                      marker=dict(size=10, color='red'))
    st.plotly_chart(fig, use_container_width=True)

    pie = clustered_df['Cluster'].value_counts().sort_index()
    fig_pie = go.Figure(go.Pie(labels=pie.index, values=pie.values))
    st.plotly_chart(fig_pie, use_container_width=True)
