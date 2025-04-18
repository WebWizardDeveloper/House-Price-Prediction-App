import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")   
    return df

df = load_data()

st.title("🏡 House Price Prediction App")

# Select some key features
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Full Bath', 'Year Built']
X = df[features]
y = df['SalePrice']

# Train model
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model(X, y)

# User input section
st.sidebar.header("Input House Features")

def user_input():
    OverallQual = st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5)
    GrLivArea = st.sidebar.slider('Above Ground Living Area (sq ft)', 500, 4000, 1500)
    GarageCars = st.sidebar.slider('Garage Capacity (cars)', 0, 4, 2)
    TotalBsmtSF = st.sidebar.slider('Basement Area (sq ft)', 0, 3000, 1000)
    FullBath = st.sidebar.slider('Bathrooms (full)', 0, 4, 2)
    YearBuilt = st.sidebar.slider('Year Built', 1900, 2022, 2000)
    data = {
        'Overall Qual': OverallQual,
        'Gr Liv Area': GrLivArea,
        'Garage Cars': GarageCars,
        'Total Bsmt SF': TotalBsmtSF,
        'Full Bath': FullBath,
        'Year Built': YearBuilt
    }
    return pd.DataFrame([data])

input_df = user_input()

# Display prediction
if st.button("Predict House Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")

# Graphs: Feature Importance
st.subheader("🔑 Feature Importance")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], align="center")
ax.set_yticks(range(len(indices)))
ax.set_yticklabels(np.array(features)[indices])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')
st.pyplot(fig)

# Graph: Correlation between Gr Liv Area and SalePrice
st.subheader("📊 Correlation between Gr Liv Area and SalePrice")

fig2, ax2 = plt.subplots()
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'], ax=ax2)
ax2.set_title('Gr Liv Area vs SalePrice')
ax2.set_xlabel('Above Ground Living Area (sq ft)')
ax2.set_ylabel('Sale Price ($)')
st.pyplot(fig2)
