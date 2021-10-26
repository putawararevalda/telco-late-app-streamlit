import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

def summarize_categoricals(df, show_levels=False):
    """
        Display uniqueness in each column
    """
    data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum()] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                           columns=['Levels', 'No. of Levels', 'No. of Missing Values'])
    return df_temp.iloc[:, 0 if show_levels else 1:]


def find_categorical(df, cutoff=10):
    """
        Function to find categorical columns in the dataframe.
    """
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) <= cutoff:
            cat_cols.append(col)
    return cat_cols


def to_categorical(columns, df):
    """
        Converts the columns passed in `columns` to categorical datatype
    """
    for col in columns:
        df[col] = df[col].astype('string')
    return df

separator = '''
---
'''

st.write("""
# Customer Late Payment Prediction App
## Created by : [Revalda Putawara](https://github.com/putawararevalda)
This app predicts the **Customer Late Payment** at a Telco Company!
Data obtained from one anonymous company in Indonesia.
""")

st.markdown(separator)

showvarinfo = st.checkbox('Show Variable Info',value=False)

if showvarinfo :
    st.markdown('The data set includes information about:\n\n\
    - Customers who left within the last month – the column is called Churn\n\
    - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies\n\
    - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges\n\
    - Demographic info about customers – gender, age range, and if they have partners and dependent')

st.markdown(separator)

showverhist = st.checkbox('Show Version History',value=False)

if showverhist:
    st.markdown('Version History of this app is as follows:\n\n\
        - Version 0.0.0 | 26/10/2021 : Initial Commit\n\
        - Version 0.1.0 : \n\
        - Version 0.1.1 : \n\
        - Version 0.1.2 : ')

st.markdown(separator)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/putawararevalda/telco-churn-app-streamlit/main/Telco-Customer-Churn-example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:

    def user_input_features():
        PSI_LATE_SC = st.sidebar.number_input('PSI_LATE_SC', min_value=0.0, value=0.0, max_value=1.0)
        LENGTH_OF_STAY = st.sidebar.number_input('LENGTH_OF_STAY', min_value=0.0, value=28.0)
        PAYMENT_inet = st.sidebar.number_input('PAYMENT_inet', min_value=0.0, value=341000.0)
        TOTAL_DURASI_inet = st.sidebar.number_input('TOTAL_DURASI_inet', min_value=0.0, value=2406595.333333333)
        TOTAL_FREQ_inet = st.sidebar.number_input('TOTAL_FREQ_inet', min_value=0.0, value=13.833333333333334)
        TOTAL_USAGE_inet = st.sidebar.number_input('TOTAL_USAGE_inet', min_value=0.0, value=325596.6666666667)
        POTS_EXIST = st.sidebar.selectbox('POTS_EXIST', (True, False))
        DUREE_ALL_pots = st.sidebar.number_input('DUREE_ALL_pots', min_value=0.0, value=0.0)
        CALL_ALL_pots = st.sidebar.number_input('CALL_ALL_pots', min_value=0.0, value=0.0)

        data = {'PSI_LATE_SC': PSI_LATE_SC,
                'LENGTH_OF_STAY': LENGTH_OF_STAY,
                'PAYMENT_inet': PAYMENT_inet,
                'TOTAL_DURASI_inet': TOTAL_DURASI_inet,
                'TOTAL_FREQ_inet': TOTAL_FREQ_inet,
                'TOTAL_USAGE_inet': TOTAL_USAGE_inet,
                'POTS_EXIST': POTS_EXIST,
                'DUREE_ALL_pots': DUREE_ALL_pots,
                'CALL_ALL_pots': CALL_ALL_pots}

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()    
    
    
# Displays the user input features
st.subheader('User Input features')

st.write(input_df.transpose())

### DATA TRANSFORMATION

telco_df = pd.read_pickle("train_data.pkl")
df = pd.concat([input_df,telco_df],axis=0)

df_1 = to_categorical(find_categorical(df), df)

# create x

# Categorical and numerical column for transforming purpise
x = df_1.copy()

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


x_trans = joblib.load('X_trans_scaler.gz')

x_encoded = x_trans.fit_transform(x)

# Generate x_encoded_input as an encoder or standardization of input data
x_encoded_data = x_encoded[1:]
x_encoded_input = x_encoded[:1]

# Reads in saved classification model
load_clf = pickle.load(open('telco-latepay-xgb.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(x_encoded_input)
prediction_proba = load_clf.predict_proba(x_encoded_input)

# Display Prediction based on input
st.subheader('Prediction')
cust_cat = np.array(['Diligent Payer','Late Payer'])
st.write(cust_cat[prediction])

# Display Prediction Probability based on input
st.subheader('Prediction Probability')
st.write(prediction_proba)
