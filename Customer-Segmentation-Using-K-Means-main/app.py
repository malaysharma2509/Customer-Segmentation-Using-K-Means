from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the model and data
filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")

# Set background color and title
st.markdown('<style>body{background-color: #f0f0f0;}</style>', unsafe_allow_html=True)
st.title("Customer Cluster Prediction")

# Create form for user input
with st.form("my_form"):
    st.header("Enter Customer Information:")
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Freqency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance, 
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    submitted = st.form_submit_button("Submit")

# Display cluster predictions and visualization if form submitted
if submitted:
    clust = loaded_model.predict(data)[0]
    st.subheader('Customer Belongs to Cluster:')
    st.write(clust)

    cluster_df1 = df[df['Cluster'] == clust]
    st.subheader('Customer Data in the Cluster:')
    st.write(cluster_df1.head())  # Check if data is available

    # Plot histograms for each feature
    plt.rcParams["figure.figsize"] = (12, 8)
    for c in cluster_df1.drop(['Cluster'], axis=1).columns:
        fig, ax = plt.subplots()
        sns.histplot(cluster_df1[c], kde=True)
        plt.title(f'{c} Distribution in Cluster {clust}')
        plt.xlabel(c)
        plt.ylabel('Frequency')
        st.pyplot(fig)

    # Plot pairplot for all features
    st.subheader('Pairplot of Features in the Cluster:')
    pairplot_fig = sns.pairplot(cluster_df1.drop(['Cluster'], axis=1))
    st.pyplot(pairplot_fig)
