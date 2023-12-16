#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
import sys, os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title('Addressing the Opioid Epidemic: A National Public Health Challenge')

@st.cache_data
def load_data():
    data = pd.read_csv(r"Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv")
    return data

d_crime = load_data()

st.write(d_crime.head(5))


# st.write(d_crime.shape)
# d_crime.head(1)


date = pd.to_datetime(d_crime['Date'])
# print(date.min())
# print(date.max())


# Calculate timedelta in days
t_delta_days = (date - date.min()) / pd.Timedelta(days=1)

d_crime['days'] = t_delta_days.astype(int)
d_crime.head(1)


def plotdata(data, cat):
    l = data.groupby(cat).size()
    l = np.log(l)
    l = l.sort_values()
    fig = plt.figure(figsize=(10, 5))
    plt.yticks(fontsize=8)
    l.plot(kind='bar', fontsize=12, color='r')
    plt.xlabel('')
    plt.ylabel('Number of reports', fontsize=10)


plotdata(d_crime, 'Category')

l = d_crime.groupby('Descript').size()
l = l.sort_values()
print(l.shape)


# Since there's 915 different crime types, let's slice by percentile and filter the crimes below 97th percentile. Let's have a look at the top types of crime for each PdDistrict.

def types_districts(d_crime, per):
    # Group by crime type and district
    hoods_per_type = d_crime.groupby('Descript').PdDistrict.value_counts(sort=True)
    t = hoods_per_type.unstack().fillna(0)

    # Sort by hood sum
    hood_sum = t.sum(axis=0)
    hood_sum = hood_sum.sort_values(ascending=False)
    t = t[hood_sum.index]

    # Filter by crime per district
    crime_sum = t.sum(axis=1)
    crime_sum = crime_sum.sort_values()

    # Large number, so let's slice the data.
    p = np.percentile(crime_sum, per)
    ix = crime_sum[crime_sum > p]
    t = t.loc[ix.index]
    return t


t = types_districts(d_crime, 97)

# Cluster the non-normalized data across the top percentile reports and each PdDistrict.
sns.clustermap(t, cmap="BuPu")

# Normalize vertically across PdDistrict.
sns.clustermap(t, standard_scale=1, vmin=0.0002, vmax=1)

# Normalize horizontally across crime types.
sns.clustermap(t, standard_scale=0)


plotdata(d_crime, 'Category')


# Let's drill down onto one
cat = d_crime[d_crime['Category'] == 'DRUG/NARCOTIC']
c = cat['Descript'].value_counts()
c.sort_values(ascending=False)
c.head(10)

# We can use what we had above, but we simply slice the input data on a category first (above).
t = types_districts(cat, 70)

sns.clustermap(t, cmap='BuPu')

sns.clustermap(t, standard_scale=1)

sns.clustermap(t, standard_scale=0)

# Let's drill down onto one
cat = d_crime[d_crime['Category'] == 'DRUG/NARCOTIC']

# Bin crime by 30 day window
cat['Month'] = np.floor(cat['days'] / 30)  # Approximate month (30 day window)

# Default
district = 'All'


def timeseries(dat, per):
    ''' Category grouped by month '''

    # Group by crime type and district 
    cat_per_time = dat.groupby('Month').Descript.value_counts(sort=True)
    t = cat_per_time.unstack().fillna(0)

    # Filter by crime per district
    crime_sum = t.sum(axis=0)
    crime_sum.sort_values()

    # Large number, so let's slice the data.
    p = np.percentile(crime_sum, per)
    ix = crime_sum[crime_sum > p]
    t = t[ix.index]
    return t


t_all = timeseries(cat, 0)

# Lets use real dates for plotting
days_from_start = pd.Series(t_all.index * 30).astype('timedelta64[D]')
dates_for_plot = date.min() + days_from_start
time_labels = dates_for_plot.map(lambda x: str(x.year) + '-' + str(x.month))


# def drug_analysis(t, district, plot):
#     fig = plt.figure(figsize=(15, 10))
#     t['BARBITUATES'] = t[map(lambda s: s.strip(), barituate_features)].sum(axis=1)
#     t['HEROIN'] = t[map(lambda s: s.strip(), heroin_features)].sum(axis=1)
#     t['HALLUCINOGENIC'] = t[map(lambda s: s.strip(), hallu_features)].sum(axis=1)
#     t['METH'] = t[map(lambda s: s.strip(), meth_features)].sum(axis=1)
#     t['WEED'] = t[map(lambda s: s.strip(), weed_features)].sum(axis=1)
#     t['COKE'] = t[map(lambda s: s.strip(), coke_features)].sum(axis=1)
#     t['METHADONE'] = t[map(lambda s: s.strip(), metadone_features)].sum(axis=1)
#     t['CRACK'] = t[map(lambda s: s.strip(), crack_features)].sum(axis=1)
#     drugs = t[['HEROIN', 'HALLUCINOGENIC', 'METH', 'WEED', 'COKE', 'CRACK']]
#     if plot:
#         drugs.index = [int(i) for i in drugs.index]
#         colors = plt.cm.jet(np.linspace(0, 1, drugs.shape[1]))
#         drugs.plot(kind='bar', stacked=True, figsize=(20, 8), color=colors, width=1, title=district, fontsize=6)
#     return drugs



# def drug_analysis_rescale(t, district, plot):
#     t['BARBITUATES'] = t[map(lambda s: s.strip(), barituate_features)].sum(axis=1)
#     t['HEROIN'] = t[map(lambda s: s.strip(), heroin_features)].sum(axis=1)
#     t['HALLUCINOGENIC'] = t[map(lambda s: s.strip(), hallu_features)].sum(axis=1)
#     t['METH'] = t[map(lambda s: s.strip(), meth_features)].sum(axis=1)
#     t['WEED'] = t[map(lambda s: s.strip(), weed_features)].sum(axis=1)
#     t['COKE'] = t[map(lambda s: s.strip(), coke_features)].sum(axis=1)
#     t['METHADONE'] = t[map(lambda s: s.strip(), metadone_features)].sum(axis=1)
#     t['CRACK'] = t[map(lambda s: s.strip(), crack_features)].sum(axis=1)
#     drugs = t[['HEROIN', 'HALLUCINOGENIC', 'METH', 'WEED', 'COKE', 'CRACK']]
#     if plot:
#         drugs = drugs.div(drugs.sum(axis=1), axis=0)
#         drugs.index = [int(i) for i in drugs.index]
#         colors = plt.cm.GnBu(np.linspace(0, 1, drugs.shape[1]))
#         colors = plt.cm.jet(np.linspace(0, 1, drugs.shape[1]))
#         drugs.plot(kind='bar', stacked=True, figsize=(20, 5), color=colors, width=1, title=district, legend=False)
#         plt.ylim([0, 1])
#     return drugs


# def real_dates():
#     # let's add the real dates
#     dates_for_plot.index = dates_for_plot
#     fig = plt.figure(figsize=(12, 8))
#     for d, c in zip(['METH', 'CRACK', 'HEROIN', 'WEED'], ['b', 'r', 'c', 'g']):
#         plt.plot(dates_for_plot.index, drug_df_all[d], 'o-', color=c, ms=6, mew=1.5, mec='white', linewidth=0.5,
#                  label=d, alpha=0.75)
#     plt.legend(loc='upper left', scatterpoints=1, prop={'size': 8})


# Filtering data for Drugs and Narcotics incidents
drugs_narcotics_data = cat[cat['Category'] == 'DRUG/NARCOTIC']

# ### Countplot of incidents per year

plt.figure(figsize=(10, 6))
sns.countplot(x=drugs_narcotics_data['Date'].str.slice(0, 4))
plt.title('Count of Drugs/Narcotics Incidents per Year')
plt.xlabel('Year')
plt.ylabel('Incident Count')
plt.xticks(rotation=45)
plt.show()

# ### Monthly trends over the years
drugs_narcotics_data['Date'] = pd.to_datetime(drugs_narcotics_data['Date'])
drugs_narcotics_data.set_index('Date', inplace=True)
monthly_count = drugs_narcotics_data.resample('M').size()

plt.figure(figsize=(10, 6))
monthly_count.plot()
plt.title('Monthly Trends of Drugs/Narcotics Incidents')
plt.xlabel('Date')
plt.ylabel('Incident Count')
plt.show()

# ### Day-wise distribution of incidents

plt.figure(figsize=(10, 6))
drugs_narcotics_data['DayOfWeek'] = drugs_narcotics_data.index.day_name()
sns.countplot(x=drugs_narcotics_data['DayOfWeek'], order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Day-wise Distribution of Drugs/Narcotics Incidents')
plt.xlabel('Day of the Week')
plt.ylabel('Incident Count')
plt.show()

# ### Top 10 districts with the highest number of incidents

top_districts = drugs_narcotics_data['PdDistrict'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_districts.plot(kind='bar')
plt.title('Top 10 Districts with Drugs/Narcotics Incidents')
plt.xlabel('District')
plt.ylabel('Incident Count')
plt.xticks(rotation=45)
plt.show()

# ### Incident resolution types

plt.figure(figsize=(8, 6))
sns.countplot(y=drugs_narcotics_data['Resolution'])
plt.title('Types of Resolutions for Drugs/Narcotics Incidents')
plt.xlabel('Incident Count')
plt.ylabel('Resolution Type')
plt.show()

# ### Heatmap showing monthly incident counts over the years

monthly_incidents = drugs_narcotics_data.groupby(
    [drugs_narcotics_data.index.year, drugs_narcotics_data.index.month]).size().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(monthly_incidents, cmap='viridis', linecolor='white', linewidth=1)
plt.title('Monthly Heatmap of Drugs/Narcotics Incidents')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

# ### Pie chart of incident categories

plt.figure(figsize=(8, 8))
drugs_narcotics_data['Descript'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
plt.title('Top 10 Incident Categories for Drugs/Narcotics')
plt.ylabel('')
plt.show()

# ### Duration between reported and occurred dates

drugs_narcotics_data['Reported_Duration'] = pd.to_datetime(
    drugs_narcotics_data.index) - pd.to_datetime(drugs_narcotics_data['Time'])
plt.figure(figsize=(10, 6))
sns.histplot(drugs_narcotics_data['Reported_Duration'].dt.days, bins=30, kde=True)
plt.title('Duration between Reported and Occurred Dates for Drugs/Narcotics Incidents')
plt.xlabel('Duration (days)')
plt.ylabel('Frequency')
plt.show()

# ### Incidents by resolution and day of the week

plt.figure(figsize=(12, 6))
sns.countplot(x='Resolution', hue='DayOfWeek', data=drugs_narcotics_data)
plt.title('Incidents by Resolution and Day of the Week')
plt.xlabel('Resolution')
plt.ylabel('Incident Count')
plt.legend(title='Day of the Week')
plt.xticks(rotation=45)
plt.show()

# # Data Preprocessing

# Handling missing values if any
missing_values = drugs_narcotics_data.isnull().sum()
print("Missing Values:\n", missing_values)

# drugs_narcotics_data123 = drugs_narcotics_data.copy()
# drugs_narcotics_data = drugs_narcotics_data123.copy()


drugs_narcotics_data['Drug_Category'] = ""
a = d_crime['Category'].apply(lambda x: 1 if x == 'DRUG/NARCOTIC' else 0)

len(np.where(a[:117821] == 1)[0])

drugs_narcotics_data['Drug_Category'] = list(a[:117821])

# Handling missing values if any
missing_values = drugs_narcotics_data.isnull().sum()
print("Missing Values:\n", missing_values)

# Dropping irrelevant columns for analysis
irrelevant_cols = ['IncidntNum', 'Descript', 'Resolution', 'Address', 'location', 'PdId']
drugs_narcotics_data.drop(columns=irrelevant_cols, inplace=True)

drugs_narcotics_data = drugs_narcotics_data.rename_axis('Date').reset_index()

# Handling date/time columns
drugs_narcotics_data['Date'] = pd.to_datetime(drugs_narcotics_data['Date'])
drugs_narcotics_data['Time'] = pd.to_datetime(drugs_narcotics_data['Time'])
drugs_narcotics_data['DayOfWeek'] = drugs_narcotics_data['Date'].dt.day_name()

# # Feature Engineering

# Extracting year, month, day, and hour from the Date column
drugs_narcotics_data['Year'] = drugs_narcotics_data['Date'].dt.year
drugs_narcotics_data['Month'] = drugs_narcotics_data['Date'].dt.month
drugs_narcotics_data['Day'] = drugs_narcotics_data['Date'].dt.day
drugs_narcotics_data['Hour'] = drugs_narcotics_data['Time'].dt.hour


# Creating a new feature for time of the day (Morning, Afternoon, Evening, Night)
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


drugs_narcotics_data['TimeOfDay'] = drugs_narcotics_data['Hour'].apply(time_of_day)

# Encoding categorical variables
drugs_narcotics_data = pd.get_dummies(drugs_narcotics_data, columns=['DayOfWeek', 'TimeOfDay', 'PdDistrict'])

# Scaling numerical columns
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# drugs_narcotics_data[['Numeric_Column1', 'Numeric_Column2']] = scaler.fit_transform(drugs_narcotics_data[['Numeric_Column1', 'Numeric_Column2']])


# Displaying the processed and engineered dataset
drugs_narcotics_data.head()

# drugs_narcotics_data.shape

# Convert TimeDelta column to numeric (days, seconds, etc.)
drugs_narcotics_data['Reported_Duration'] = drugs_narcotics_data['Reported_Duration'].dt.days

drugs_narcotics_data.dropna(inplace=True, axis=1)
# drugs_narcotics_data.shape


def convert_cat_to_numeric(df):
    """Converts all categorical columns to numeric in a Pandas DataFrame."""
    for col in df.select_dtypes(include='category'):
        df[col] = df[col].cat.codes
    return df


drugs_narcotics_data = convert_cat_to_numeric(drugs_narcotics_data)

### Smote


# Select rows with value 1 in "category"
df_filtered_1 = drugs_narcotics_data[drugs_narcotics_data['Drug_Category'] == 1]

# Select the same number of rows with value 0 in "category"
df_filtered_0 = drugs_narcotics_data[drugs_narcotics_data['Drug_Category'] == 0][:len(df_filtered_1)]

# Combine the two DataFrames
drugs_narcotics_dataa = pd.concat([df_filtered_1, df_filtered_0])

# ![Picture10.png](attachment:Picture10.png)


# Features and target variable
features = drugs_narcotics_data.drop(columns=['Drug_Category', 'Category', 'Date', 'Time'])
target = drugs_narcotics_data['Drug_Category']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# # Model Training and Prediction
# @st.cache_data(allow_output_mutation=True)
import pickle

def train_models(X_train, X_test, y_train, y_test, logistic_reg_params, knn_params, rf_params, xgb_params):
    models = {
        'Logistic Regression': LogisticRegression(**logistic_reg_params),
        'KNN': KNeighborsClassifier(**knn_params),
        'Random Forest': RandomForestClassifier(**rf_params),
        'XGBoost': XGBClassifier(**xgb_params)
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        with open(name+'.pkl', 'wb') as file:
            pickle.dump(model, file)
        predictions[name] = {
            'model': model,
            'y_pred': y_pred
        }
    return predictions


# Streamlit app UI
st.title('Model Training and Evaluation')

# Options to adjust model parameters
st.sidebar.subheader('Model Parameters')
logistic_reg_params = {
    'max_iter': st.sidebar.slider('Logistic Regression - Max Iterations', 100, 1000, 100)
}

knn_params = {
    'n_neighbors': st.sidebar.slider('KNN - Number of Neighbors', 1, 20, 5)
}

rf_params = {
    'n_estimators': st.sidebar.slider('Random Forest - Number of Estimators', 10, 200, 100),
    'random_state': 42
}

xgb_params = {
    'n_estimators': st.sidebar.slider('XGBoost - Number of Estimators', 10, 200, 100),
    'random_state': 42
}

# Display current parameter values
st.sidebar.subheader('Current Parameter Values')
st.sidebar.write("Logistic Regression:", logistic_reg_params)
st.sidebar.write("KNN:", knn_params)
st.sidebar.write("Random Forest:", rf_params)
st.sidebar.write("XGBoost:", xgb_params)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Function call to train models
models_predictions = train_models(X_train, X_test, y_train, y_test, logistic_reg_params, knn_params, rf_params,
                                  xgb_params)



# Display the model evaluation metrics
st.subheader('Model Evaluation Metrics')
for name, prediction_data in models_predictions.items():
    model = prediction_data['model']
    y_pred = prediction_data['y_pred']

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**{name}**")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    st.write("\n")


# Function to plot ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, prediction_data in models.items():
        model = prediction_data['model']

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=0.8)

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()


# Display ROC curves
st.subheader('ROC Curves for Models')
roc_plot = plot_roc_curves(models_predictions, X_test, y_test)
st.pyplot(roc_plot)
