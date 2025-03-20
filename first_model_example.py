import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# Streamlit UI
st.title(":bar_chart: Insider Trading Anomaly Detection")
st.write("Upload a stock financial dataset to detect anomalies using Machine Learning.")
# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Drop unnecessary columns
    columns_to_drop = ["gvkey", "datadate", "indfmt", "consol", "popsrc", "datafmt", "curcd", "costat", "ugi", "urect"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # Convert fyear to datetime format
    df['fyear'] = pd.to_datetime(df['fyear'], format='%Y', errors='coerce')
    # Feature Engineering
    df['profit_margin'] = df['ni'] / df['revt']
    df['leverage_ratio'] = df['lt'] / df['at']
    df['asset_turnover'] = df['revt'] / df['at']
    df['sga_to_revenue'] = df['xsga'] / df['revt']
    # Handling missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Normalize Data
    scaler = StandardScaler()
    numeric_cols = ["at", "cogs", "dp", "gp", "invch", "lt", "ni", "ppegt", "revt", "xsga",
                    "profit_margin", "leverage_ratio", "asset_turnover", "sga_to_revenue"]
    df_scaled = scaler.fit_transform(df[numeric_cols])
    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly_if'] = iso_forest.fit_predict(df_scaled)
    # Apply One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    df['anomaly_svm'] = oc_svm.fit_predict(df_scaled)
    st.subheader("Anomaly Detection Results")
    st.write(df[['tic', 'fyear', 'profit_margin', 'leverage_ratio', 'anomaly_if', 'anomaly_svm']])
    # Visualization for Isolation Forest
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=df['fyear'], y=df['profit_margin'], hue=df['anomaly_if'], palette={1: 'blue', -1: 'red'}, ax=ax)
    ax.set_title("Isolation Forest: Anomalies in Profit Margin Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Profit Margin")
    st.pyplot(fig)
    # Visualization for One-Class SVM
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=df['fyear'], y=df['leverage_ratio'], hue=df['anomaly_svm'], palette={1: 'blue', -1: 'red'}, ax=ax)
    ax.set_title("One-Class SVM: Anomalies in Leverage Ratio Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Leverage Ratio")
    st.pyplot(fig)
st.sidebar.info("Supported File Format: CSV")
