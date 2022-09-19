import pandas as pd
import numpy as np
import streamlit as st

import warnings 
import plotly.express as px
import shap
warnings.simplefilter('ignore')

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

DATA_URL = "train.csv"
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
df = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
feature_lst = ['delivery_duration', 
               'mode_of_transport', 'no_units', 
               'cust_group_name', 'cust_segment_name', 'customer_id', 
               'type', 'item_class_l1', 'item_class_l2', 'item_class_l3',
               'colour', 'plant_nr', 'plant_city', 'plant_country_name', 'most_expensive_part_l1', 
               'most_expensive_part_l2', 'shipto_city', 'shipto_nr', 'soldto_city', 'soldto_nr', 
               'car_nr', 'shipto_country', 'soldto_country','rolling_mean_t7','rolling_mean_t30','rolling_mean_t60',
              'rolling_mean_t90','rolling_mean_t180']
X = df[feature_lst]
y = df['unit_price_k']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = XGBRegressor()
# fit model
model.fit(X_train.values, y_train)

yhat = model.predict(X_test.values)


rmse = (np.sqrt(mean_squared_error(y_test, yhat)))
r2 = r2_score(y_test, yhat)
st.write (
    "Testing performance",
"\nRMSE: {:.2f}".format(rmse),
"\nR2: {:.2f}".format(r2),
)
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
feat_imp_df = pd.DataFrame(range(len(sorted_idx)), feature_importance[sorted_idx])
feat_imp_df['features'] = np.array(X_test.columns)[sorted_idx]
feat_imp_df.reset_index(inplace = True)
st.write(
    feat_imp_df,
    )
fig = px.bar(        
        feat_imp_df,
        x = "index",
        y = "features",
        title = "Feature importance plot",
        )
st.plotly_chart(fig)

