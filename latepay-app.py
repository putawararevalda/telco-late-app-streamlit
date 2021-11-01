import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
import joblib
from xgboost import XGBClassifier
from PIL import Image


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


st.set_page_config(page_title="Late Pay Prediction App", page_icon="💰")

separator = '''
---
'''

#image=Image.open('logo-telkomathon.jpg')

#st.image(image, use_column_width=True)

vidfilename = "telkomathon-vid.mp4"
video_file = open(vidfilename, 'rb')
video_bytes = video_file.read()


st.header("Telkomathon Batch 2")
st.sidebar.video(video_bytes)


st.write("""
# Customer Late Payment Prediction App
## Created by : [Revalda Putawara](https://github.com/putawararevalda)
This app predicts the **Customer Late Payment** at a Telco Company!
Data obtained from one anonymous company in Indonesia.
""")

image=Image.open('logo-telkomathon.jpg')
st.sidebar.image(image, use_column_width=True)

st.markdown(separator)

showvarinfo = st.checkbox('Show TUTORIAL',value=False)

if showvarinfo :

    st.write("""
    ## Aplikasi ini merupakan model *MACHINE LEARNING* yang mengolah input data customer dan memprediksikan kemungkinan pelanggan tersebut terlambat membayar tagihan pada bulan depan.

    #### Variabel data customer yang digunakan adalah:
    1. **M1 - M6 payment status** : Terlambat / tidaknya pelanggan membayar tagihan internet per bulannya selama 6 bulan terakhir. (skema urutan bulan dapat dilihat pada bagian bawah dalam section ini)
    \n M1 - M6 payment status akan dikonversikan menjadi nilai PSI_LATE_SC (range nilai 0 - 1).\n
    2. **Length of Stay** : Lama berlangganan (dalam bulan) \n
    3. **PAYMENT_inet** : Rata-rata tagihan internet yang dibayarkan pelanggan dalam 6 bulan terakhir \n
    4. **TOTAL_DURASI_inet** : Rata-rata durasi penggunaan internet dalam 6 bulan terakhir \n
    5. **TOTAL_FREQ_inet** : Rata-rata frekuensi penggunaan internet dalam 6 bulan terakhir \n
    6. **TOTAL_USAGE_inet** : Rata-rata jumlah penggunaan internet dalam 6 bulan terakhir (upload & download dalam MB)\n
    7. **POTS_EXIST** : Ada/tidaknya service telepon (POTS) pada pelanggan (True/False)
    8. **DUREE_ALL_pots** : Rata-rata durasi penggilan keluar untuk seluruh kategori panggilan selama 6 bulan terakhir \n
    9. **CALL_ALL_pots** : Rata-rata jumlah penggilan keluar untuk seluruh kategori panggilan selama 6 bulan terakhir \n\n
    Data default yang diberikan pada aplikasi ini merupakan rata-rata dari sampel yang tersedia. \n
    ### Silahkan ubah variabel yang ada untuk melihat pengaruh variabel terhadap keterlambatan pembayaran pelanggan!
    """)

    st.write("""
    ## Skema timeline pelanggan
    """)
    image2=Image.open('timeline.png')
    st.image(image2)

    st.write("""
    ## Sampel data input model
    """)

    image3=Image.open('datalist.png')
    st.image(image3)

    st.write("""
    ## The *MAGIC* starts in the next section, try it!
    """)

st.markdown(separator)

st.write("""
# FEATURE INPUT : PAST 6 MONTHS PAYMENT BEHAVIOUR
""")

st.markdown(separator)

################### MENU FOR SINGLE CUSTOMER INPUT

col1c, col2c, col3c = st.columns(3)
with col1c:
    ps1 = st.selectbox('M1 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))
with col2c:
    ps2 = st.selectbox('M2 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))
with col3c:
    ps3 = st.selectbox('M3 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))

col1d, col2d, col3d = st.columns(3)
with col1d:
    ps4 = st.selectbox('M4 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))
with col2d:
    ps5 = st.selectbox('M5 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))
with col3d:
    ps6 = st.selectbox('M6 payment status', ('NOT_LATE', 'LATE_SAME_MONTH','LATE_NEXT_MONTH'))

st.markdown(separator)


##########33 COUNT FOR PSI_LATE_SC

var1 = ['M1', 'M2', 'M3', 'M4','M5','M6']
var2 = [ps1,ps2,ps3,ps4,ps5,ps6]

var1_series = pd.Series(var1)
var2_series = pd.Series(var2)

frame_pls = {'Period': var1_series, 'Payment Status': var2_series}

pls_df = pd.DataFrame(frame_pls)

pay_status_dict = {"LATE_SAME_MONTH": 1, "LATE_NEXT_MONTH" : 1, "NOT_LATE" : 0}
pls_df['PSI_LATE_score'] = pls_df['Payment Status'].map(pay_status_dict)

psi_late_sc_score = pls_df['PSI_LATE_score'].mean()


st.write("""
# FEATURE INPUT : SERVICE RELATED VARIABLES
""")

st.markdown(separator)

col1e, col2e, col3e = st.columns(3)

with col1e:
    st.metric(label="PSI_LATE_SC", value="{0:.4}".format(psi_late_sc_score))
with col2e:
    LENGTH_OF_STAY = st.number_input('LENGTH_OF_STAY', min_value=0.0, value=28.0, step=1.0)
with col3e:
    PAYMENT_inet = st.number_input('PAYMENT_inet', min_value=0.0, value=341000.0)

col1f, col2f, col3f = st.columns(3)

with col1f:
    TOTAL_DURASI_inet = st.number_input('TOTAL_DURASI_inet', min_value=0.0, value=2406595.333333333)
with col2f:
    TOTAL_FREQ_inet = st.number_input('TOTAL_FREQ_inet', min_value=0.0, value=13.833333333333334)
with col3f:
    TOTAL_USAGE_inet = st.number_input('TOTAL_USAGE_inet', min_value=0.0, value=325596.6666666667)


col1g, col2g, col3g = st.columns(3)

with col1g:
    POTS_EXIST = st.selectbox('POTS_EXIST', (True, False))
with col2g:
    DUREE_ALL_pots = st.number_input('DUREE_ALL_pots', min_value=0.0, value=0.0)
with col3g:
    CALL_ALL_pots = st.number_input('CALL_ALL_pots', min_value=0.0, value=0.0)


### USER INPUT DATAFRAME DEFINITION

data = {'PSI_LATE_SC': psi_late_sc_score ,
'LENGTH_OF_STAY': LENGTH_OF_STAY,
'PAYMENT_inet': PAYMENT_inet,
'TOTAL_DURASI_inet': TOTAL_DURASI_inet,
'TOTAL_FREQ_inet': TOTAL_FREQ_inet,
'TOTAL_USAGE_inet': TOTAL_USAGE_inet,
'POTS_EXIST': POTS_EXIST,
'DUREE_ALL_pots': DUREE_ALL_pots,
'CALL_ALL_pots': CALL_ALL_pots}

features = pd.DataFrame(data, index=[0])

input_df = features.copy()   
    
st.markdown(separator)    

# Displays the user input features
#st.subheader('User Input features')

#st.write(input_df.transpose())



#st.write(pls_df)





### DATA TRANSFORMATION

telco_df = pd.read_pickle("train_data.pkl")
input_df = input_df[telco_df.columns.tolist()]
df = pd.concat([input_df,telco_df],axis=0)



df_1 = to_categorical(find_categorical(df), df)

# create x

# Categorical and numerical column for transforming purpise
x = df_1.copy()

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


#x_trans = joblib.load('X_trans_scaler.gz')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import RobustScaler

x_trans = ColumnTransformer(remainder='passthrough',
                  transformers=[('one_hot_encoder',
                                 OneHotEncoder(drop='first', dtype='int'),
                                 ['POTS_EXIST']),
                                ('robust_scaler', RobustScaler(),
                                 ['LENGTH_OF_STAY', 'PAYMENT_inet',
                                  'TOTAL_DURASI_inet', 'DUREE_ALL_pots',
                                  'TOTAL_FREQ_inet', 'CALL_ALL_pots',
                                  'TOTAL_USAGE_inet', 'PSI_LATE_SC'])])

x_encoded = x_trans.fit_transform(x)

# Generate x_encoded_input as an encoder or standardization of input data
x_encoded_data = x_encoded[1:]
x_encoded_input = x_encoded[:1]


# Reads in saved classification model
load_clf = pickle.load(open('telco-latepay-xgb.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(x_encoded_input)
prediction_proba = load_clf.predict_proba(x_encoded_input)

st.write("""
# PREDICTION RESULT
""")

st.markdown(separator)

col1h, col2h = st.columns(2)

with col1h:
    # Display Prediction based on input
    st.subheader('Prediction')
    cust_cat = np.array(['Diligent Payer','Late Payer'])
    st.metric(label="Prediction", value=cust_cat[prediction][0])
with col2h:
    # Display Prediction Probability based on input
    st.subheader('Probability of Paying Late')
    st.metric(label="Probability of Paying Late", value="{0:.2%}".format(prediction_proba[0][1]))

st.markdown(separator)

st.write("""
## Try changing the variable values, the probability will change!
""")

st.markdown(separator)


st.write("""
### That's the end of the *MACHINE LEARNING** part, sections below wil be related to the documentation of the model.
### See simulation section for more experimentation!
""")



st.markdown(separator)

showverhist = st.checkbox('Show Version History',value=False)

if showverhist:
    st.markdown('Version History of this app is as follows:\n\n\
        - Version 0.0.0 | 26/10/2021 : Initial Commit\n\
        - Version 0.1.0 | 27/10/2021 : Add model info, Add some cosmetics\n\
        - Version 0.1.1 : \n\
        - Version 0.1.2 : ')

st.markdown(separator)


showmodel = st.checkbox('Show Model Result',value=False)

if showmodel :
    st.header("Confusion Matrix")
    image4=Image.open('confmatrix.png')
    st.image(image4)


    st.header("Train-Test Details")
    image5=Image.open('ttdetails.png')
    st.image(image5)

    st.header("Classification Report")
    image5=Image.open('reportdetails.png')
    st.image(image5)

    st.header("Feature Importance")
    image6=Image.open('featureimp.png')
    st.image(image6)

    st.header("AUC Score for Test Data")
    image7=Image.open('roctest.png')
    st.image(image7)

st.markdown(separator)

showsim = st.checkbox('Show Simulation',value=False)

if showsim :
    custnum = st.number_input('# Customer in Indonesia', min_value=0, value=8000000, step=1)
    churn_rate = st.slider(label= "Churn Rate", min_value=0.00, max_value=1.00, value=0.70, step=0.05)

    col1a, col2a= st.columns(2)

    with col1a:
        norm_arpu = st.number_input('Normal ARPU', min_value=0, value=341000, step=1000)
    with col2a:
        down_arpu = st.number_input('Downgraded ARPU', min_value=0, value=275000, step=1000)
    
    col1b, col2b= st.columns(2)

    with col1b:
        churnprog_rate = st.slider(label= "Churn Rate Assumption (after program)", min_value=0.00, max_value=1.00, value=0.20, step=0.05)
    with col2b:
        downprog_rate = st.slider(label= "Downgrade Rate Assumption (after program)", min_value=0.00, max_value=1.00, value=0.50, step=0.05)


    act = ['Diligent', 'Late', 'Diligent', 'Late']
    pred = ['Diligent', 'Late', 'Late', 'Diligent']
    sample = [ 13064, 3016, 2984, 936]
  
    act_series = pd.Series(act)
    pred_series = pd.Series(pred)
    sample_series = pd.Series(sample)
  
    frame = {'Actual': act_series, 'Predicted': pred_series,'Sample Cust' : sample_series }
  
    result_df = pd.DataFrame(frame)

    result_df["All Cust"] = 0

    for xd in range(len(result_df)):
        result_df.at[xd,"All Cust"] = math.ceil(result_df.at[xd,"Sample Cust"]/result_df["Sample Cust"].sum()*custnum)

    result_df["Collection 0"] = 0

    result_df.at[0,"Collection 0"] = result_df.at[0,"All Cust"] * norm_arpu
    result_df.at[1,"Collection 0"] = (1-churn_rate) * result_df.at[1,"All Cust"] * norm_arpu 
    result_df.at[2,"Collection 0"] = (1-churn_rate) * result_df.at[2,"All Cust"] * norm_arpu 
    result_df.at[3,"Collection 0"] = result_df.at[3,"All Cust"] * norm_arpu

    result_df["Collection 1"] = 0

    result_df.at[0,"Collection 1"] = result_df.at[0,"All Cust"] * norm_arpu
    result_df.at[1,"Collection 1"] = (1-(churnprog_rate+downprog_rate)) * result_df.at[1,"All Cust"] * norm_arpu + downprog_rate * result_df.at[1,"All Cust"] * down_arpu
    result_df.at[2,"Collection 1"] = (1-churn_rate) * result_df.at[2,"All Cust"] * norm_arpu 
    result_df.at[3,"Collection 1"] = result_df.at[3,"All Cust"] * norm_arpu

    delta = (result_df["Collection 1"].sum() - result_df["Collection 0"].sum())/result_df["Collection 0"].sum()

    st.subheader('Simulation Result')

    st.write(result_df)


    st.header("Colection Result Comparison")

    col1, col2= st.columns(2)

    with col1:
        st.header("Scenario 0")
        st.metric(label="Collection (doing nothing)", value='Rp. {:,}'.format(result_df["Collection 0"].sum()))

    with col2:
        st.header("Scenario 1")
        st.metric(label="Collection (with program)", value='Rp. {:,}'.format(result_df["Collection 1"].sum()), delta="{0:.2%}".format(delta))

    st.header("Improvement")
    st.metric(label="Collection addition", value='Rp. {:,}'.format(result_df["Collection 1"].sum() - result_df["Collection 0"].sum()))

    
    
    




st.markdown(separator)



#st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/putawararevalda/telco-churn-app-streamlit/main/Telco-Customer-Churn-example.csv)
#""")

# Collects user input features into dataframe
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file (feature not ready)", type=["csv"])
#if uploaded_file is not None:
    #input_df = pd.read_csv(uploaded_file)
#else:

    #def user_input_features():
        #PSI_LATE_SC = st.sidebar.slider(label= "PSI_LATE_SC", min_value=0.00, max_value=1.00, value=0.00, step=1/6)
        
        
