import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
        PSI_LATE_SC = st.sidebar.number_input('PSI_LATE_SC', min_value=0.0, value=0.0, max_value=1)
        LENGTH_OF_STAY = st.sidebar.number_input('LENGTH_OF_STAY', min_value=0.0, value=28)
        PAYMENT_inet = st.sidebar.number_input('PAYMENT_inet', min_value=0.0, value=341000)
        TOTAL_DURASI_inet = st.sidebar.number_input('TOTAL_DURASI_inet', min_value=0.0, value=2406595.333333333)
        TOTAL_FREQ_inet = st.sidebar.number_input('TOTAL_FREQ_inet', min_value=0.0, value=13.833333333333334)
        TOTAL_USAGE_inet = st.sidebar.number_input('TOTAL_USAGE_inet', min_value=0.0, value=325596.6666666667)
        POTS_EXIST = st.sidebar.selectbox('POTS_EXIST', (True, False))
        DUREE_ALL_pots = st.sidebar.number_input('DUREE_ALL_pots', min_value=0.0, value=32.37)
        CALL_ALL_pots = st.sidebar.number_input('CALL_ALL_pots', min_value=0.0, value=32.37)

        data = {'gender': gender,
                'SeniorCitizen': SeniorCitizen,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity}

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

"""COMMENT FOR TRIAL PURPOSE START.

# Combines user input features with entire telco dataset
# This will be useful for the encoding phase
telco_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
telco_df = telco_raw.drop(columns=['customerID','Churn'])

df = pd.concat([input_df,telco_df],axis=0)

# totalcharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# value mapping
## Shorten the Labels
value_mapper = {'Female': 'F', 'Male': 'M', 'Yes': 'Y', 'No': 'N',
                'No phone service': 'No phone', 'Fiber optic': 'Fiber',
                'No internet service': 'No internet', 'Month-to-month': 'Monthly',
                'Bank transfer (automatic)': 'Bank transfer',
                'Credit card (automatic)': 'Credit card',
                'One year': '1 yr', 'Two year': '2 yr'}
df_1 = df.replace(to_replace=value_mapper)


# df columns to lower
df_1.columns = [label.lower() for label in df_1.columns]

# remove tenure = 0
df_1.drop(labels=df_1[df_1['tenure'] == 0].index, axis=0, inplace=True)

# convert to categorical
df_1 = to_categorical(find_categorical(df_1), df_1)

# reorder column so tenure get beside numerical features
new_order = list(df_1.columns)
new_order.insert(16, new_order.pop(4))
df_1 = df_1[new_order]

# Categorical and numerical column for transforming purpise
x = df_1.copy()
categorical_columns = list(x.select_dtypes(include='string').columns)
numeric_columns = list(x.select_dtypes(exclude='string').columns)

"""

# Displays the user input features
st.subheader('User Input features')

st.write(input_df.transpose())

"""COMMENT FOR TRIAL PURPOSE START.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

## Column Transformer
transformers = [('one_hot_encoder',
                  OneHotEncoder(drop='first',dtype='int'),
                  categorical_columns),
                ('standard_scaler', StandardScaler(), numeric_columns)]
x_trans = ColumnTransformer(transformers, remainder='passthrough')

## Applying Column Transformer
x_encoded = x_trans.fit_transform(x)

# Generate x_encoded_input as an encoder or standardization of input data
x_encoded_data = x_encoded[1:]
x_encoded_input = x_encoded[:1]

# Standard Scaling function done last, so that the input won't spoil the raw data
x_encoded_input[0,-3] = (df_1.iloc[0,-3] - df_1.iloc[1:,-3].mean()) / df_1.iloc[1:,-3].std()
x_encoded_input[0,-2] = (df_1.iloc[0,-2] - df_1.iloc[1:,-2].mean()) / df_1.iloc[1:,-2].std()
x_encoded_input[0,-1] = (df_1.iloc[0,-1] - df_1.iloc[1:,-1].mean()) / df_1.iloc[1:,-1].std()

# Reads in saved classification model
load_clf = pickle.load(open('telco-churn_logreg_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(x_encoded_input)
prediction_proba = load_clf.predict_proba(x_encoded_input)

# Display Prediction based on input
st.subheader('Prediction')
cust_cat = np.array(['No Churn','Churn'])
st.write(cust_cat[prediction])

# Display Prediction Probability based on input
st.subheader('Prediction Probability')
st.write(prediction_proba)

"""
