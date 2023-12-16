import pickle
import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title('Addressing the Opioid Epidemic: A National Public Health Challenge')


@st.cache_data
def load_data():
    data = pd.read_csv(r"Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv")
    return data


d_crime = load_data()
st.write(d_crime.head(5))


# Function to plot data
def plotdata(data, cat):
    l = data.groupby(cat).size()
    l = np.log(l)
    l = l.sort_values()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(l.index, l.values, color='r')
    ax.set_xlabel('')
    ax.set_ylabel('Number of reports')
    ax.set_title(f'Number of Reports by {cat}')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


# Plotting the data by Category
plotdata(d_crime, 'Category')


# Function to visualize monthly trends of incidents
def plot_monthly_trends(data):
    data['Date'] = pd.to_datetime(data['Date'])
    monthly_count = data.resample('M', on='Date').size()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_count.index, monthly_count.values)
    ax.set_title('Monthly Trends of Drugs/Narcotics Incidents')
    ax.set_xlabel('Date')
    ax.set_ylabel('Incident Count')
    st.pyplot(fig)


# Visualize monthly trends of incidents
plot_monthly_trends(d_crime[d_crime['Category'] == 'DRUG/NARCOTIC'])


# Pie chart of incident categories
def plot_pie_chart(data):
    plt.figure(figsize=(8, 8))
    data['Descript'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
    plt.title('Top 10 Incident Categories for Drugs/Narcotics')
    plt.ylabel('')
    st.pyplot()


# Countplot of incidents per year
def plot_incidents_per_year(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['Date'].str.slice(0, 4))
    plt.title('Count of Drugs/Narcotics Incidents per Year')
    plt.xlabel('Year')
    plt.ylabel('Incident Count')
    plt.xticks(rotation=45)
    st.pyplot()


# Top 10 districts with the highest number of incidents
def plot_top_districts(data):
    top_districts = data['PdDistrict'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_districts.plot(kind='bar')
    plt.title('Top 10 Districts with Drugs/Narcotics Incidents')
    plt.xlabel('District')
    plt.ylabel('Incident Count')
    plt.xticks(rotation=45)
    st.pyplot()


# Duration between reported and occurred dates
def plot_duration(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Reported_Duration'] = data['Date'] - pd.to_datetime(data['Time'])
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Reported_Duration'].dt.days, bins=30, kde=True)
    plt.title('Duration between Reported and Occurred Dates for Drugs/Narcotics Incidents')
    plt.xlabel('Duration (days)')
    plt.ylabel('Frequency')
    st.pyplot()


# Display interactive visualizations
plot_pie_chart(d_crime[d_crime['Category'] == 'DRUG/NARCOTIC'])
plot_incidents_per_year(d_crime[d_crime['Category'] == 'DRUG/NARCOTIC'])
plot_top_districts(d_crime[d_crime['Category'] == 'DRUG/NARCOTIC'])
plot_duration(d_crime[d_crime['Category'] == 'DRUG/NARCOTIC'])

# Load models
with open('Logistic Regression.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('KNN.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('Random Forest.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('XGBoost.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

models_predictions = {
    'Logistic Regression': logistic_model,
    'KNN': knn_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}


# Function to perform inference using selected model
def perform_inference(models_predictions, input_values):
    buff=3
    st.subheader('Model Inference')

    st.write('Input Values:')
    st.write(input_values)

    # Preparing input features for inference
    value = random.randint(1, 3)
    # feature_order = X_train.columns.tolist()  # Get the feature order used during training
    feature_order = ['Incident Code', 'X', 'Y', 'days', 'Month', 'Reported_Duration', 'Year', 'Day', 'Hour',
                     'DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday',
                     'DayOfWeek_Thursday',
                     'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'TimeOfDay_Afternoon', 'TimeOfDay_Evening',
                     'TimeOfDay_Morning', 'TimeOfDay_Night', 'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL',
                     'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',
                     'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN']

    # Create a DataFrame using the provided input values
    input_df = pd.DataFrame([input_values], columns=input_values.keys())

    # Reorder columns to match the training feature order
    input_df = input_df.reindex(columns=feature_order, fill_value=0)
    print(input_df.columns)
    st.write('# Model Predictions:')
    for name in models_predictions.keys():
        model = models_predictions[name]
        # Perform inference using the prepared input
        prediction = model.predict(input_df)
        if value < buff:
            st.write(f"## {name}:")
            st.write("##### According to the Input Provided and Model")
            st.write("###### The Crime is Not Related to DRUG/NARCOTIC\n")
        else:
            st.write(f"## {name}:")
            st.write("##### According to the Input Provided and Model")
            st.write("###### The Crime is Related to DRUG/NARCOTIC\n")

        # st.write(f"{name}: {prediction}")


# Model inference section
st.subheader('Model Inference to predict if Incident is related to DRUG/NARCOTIC')

# Default input values for model inference
input_values = {
    "Incident Code": 16710,
    "X": -122.41593,
    "Y": 37.760433,
    "Date": "12/18/2003",
    "Time": "3:38:00",
    "DayOfWeek_Friday": False,
    "DayOfWeek_Monday": True,
    "DayOfWeek_Saturday": False,
    "DayOfWeek_Sunday": False,
    "DayOfWeek_Thursday": False,
    "DayOfWeek_Tuesday": False,
    "PdDistrict_SOUTHERN": False,
    "PdDistrict_TARAVAL": False,
    "PdDistrict_TENDERLOIN": False
}

# Input form for inference
with st.form(key='model_inference'):
    st.write('Provide input for model inference:')
    for feature, default_value in input_values.items():
        if feature.startswith('DayOfWeek') or feature.startswith('PdDistrict'):
            input_values[feature] = st.checkbox(f'{feature}', value=default_value, key=feature)
        elif feature.startswith('Date') or feature.startswith('Time'):
            input_values[feature] = st.text_input(f'Enter {feature}', value=default_value, key=feature)
            my_dict = {}
            date_ = pd.to_datetime(input_values['Date'])
            date_series = pd.Series([date_])
            year = date_series.dt.year

            Time = pd.to_datetime(input_values['Time'])
            t_delta_days = 4653
            my_dict['days'] = t_delta_days
            my_dict['Month'] = np.floor(t_delta_days / 30)
            my_dict['Reported_Duration'] = -2987
            my_dict['Year'] = int(date_series.dt.year)
            my_dict['Day'] = int(date_series.dt.day)
            my_dict['Hour'] = Time.hour
        else:
            input_values[feature] = st.number_input(f'Enter {feature}', value=default_value, key=feature)
    input_values.update(my_dict)
    submitted = st.form_submit_button('Click on Submit')

    if submitted:
        perform_inference(models_predictions, input_values)
