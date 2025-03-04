import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import shap
import datetime as dt

# Set Page Config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load trained model
with open("best_fraud_detection_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load Data
@st.cache_data
def load_data():
    crd = pd.read_csv("dataset/fraudTrain.csv")
    crd['trans_date_trans_time'] = pd.to_datetime(crd['trans_date_trans_time'])
    crd['hour'] = crd['trans_date_trans_time'].dt.hour
    crd['day'] = crd['trans_date_trans_time'].dt.dayofweek
    crd['month'] = crd['trans_date_trans_time'].dt.month
    crd['age'] = dt.date.today().year - pd.to_datetime(crd['dob']).dt.year
    return crd

data = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Fraud Prediction", "Dashboard Analytics"])

if page == "Fraud Prediction":
    st.sidebar.subheader("Fraud Detection Panel")
    st.title("💳 Credit Card Fraud Prediction")
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("💰 Transaction Amount", min_value=0.01, format="%.2f")
        city_pop = st.number_input("🏩 City Population", min_value=1)
        age = st.number_input("🎂 Customer Age", min_value=18, max_value=100)
        lat_dist = st.number_input("📍 Latitudinal Distance", format="%.6f")

    with col2:
        long_dist = st.number_input("📍 Longitudinal Distance", format="%.6f")
        hour = st.slider("🕒 Transaction Hour", 0, 23, 12)
        day = st.slider("🗓 Transaction Day", 1, 31, 15)
        month = st.slider("📆 Transaction Month", 1, 12, 6)

    st.markdown("### 📂 Transaction Category")
    category_options = [
        "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
        "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
        "shopping_net", "shopping_pos", "travel"
    ]
    selected_category = st.selectbox("Select Category", category_options)
    category_dict = {f"category_{cat}": 0 for cat in category_options}
    category_dict[f"category_{selected_category}"] = 1

    st.markdown("### 💻 Customer Gender")
    gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
    gender_M = 1 if "Male" in gender else 0

    input_features = [
        amt, city_pop, age, hour, day, month, lat_dist, long_dist,
        *category_dict.values(), gender_M
    ]

    input_array = np.array(input_features).reshape(1, -1)

    if st.button(" Predict Fraud"):
        fraud_probability = model.predict_proba(input_array)[0][1]
        fraud_risk_score = round(fraud_probability * 100, 2)

        # Define risk levels
        if fraud_risk_score < 30:
            risk_level = "🟢 Low Risk"
            st.success(f"✅ The transaction is NOT fraudulent. Risk Score: {fraud_risk_score}% ({risk_level})")
        elif fraud_risk_score < 70:
            risk_level = "🟠 Medium Risk"
            st.warning(f"⚠ Suspicious transaction. Risk Score: {fraud_risk_score}% ({risk_level})")
        else:
            risk_level = "🔴 High Risk"
            st.error(f"🚨 The transaction is FRAUDULENT! Risk Score: {fraud_risk_score}% ({risk_level})")

        st.progress(fraud_risk_score / 100)

        # SHAP Explainability
        st.subheader("🔍 Feature Contributions to Fraud Risk")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_array)

        # Extract the first prediction's explanation for the fraud class
        shap_explanation = shap_values[0,:,1]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(shap_explanation, max_display=10, show=False)
        

        # Manually label the axes with column names
        feature_names = [
            "amt", "city_pop", "age", "hour", "day", "month", "lat_dist", "long_dist",
            "category_food_dining", "category_gas_transport", "category_grocery_net",
            "category_grocery_pos", "category_health_fitness", "category_home",
            "category_kids_pets", "category_misc_net", "category_misc_pos",
            "category_personal_care", "category_shopping_net", "category_shopping_pos",
            "category_travel", "gender_M"
        ]
        top_features = feature_names[:10][::-1] # Get top 10 and reverse
        ax.set_yticks(np.arange(len(top_features) + 1)) # Set ticks to match labels
        ax.set_yticklabels([""] + top_features) # Add labels
        st.pyplot(fig)


elif page == "Dashboard Analytics":
    st.sidebar.subheader("Analytics Panel")
    st.title("📊 Fraud Detection Dashboard")

    selected_plot = st.sidebar.selectbox("Select a visualization", [
        "Transaction Amount Distribution",
        "Fraud vs. Non-Fraud Transactions by Gender",
        "Fraud Distribution by Transaction Category",
        "Fraud Percentage by Age Group",
        "Fraud Percentage by Hour of Day",
        "Fraud Percentage by Day of Week",
        "Fraud Percentage by Month",
        "Fraudulent Transactions by State"
    ])

    if selected_plot == "Transaction Amount Distribution":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='is_fraud', y='amt', data=data, ax=ax)
        ax.set_xlabel("Fraud Status")
        ax.set_ylabel("Transaction Amount (USD)")
        st.pyplot(fig)
        st.write("This box plot shows the spread of transaction amounts by fraud status. Higher outliers may indicate high-risk transactions.")
        st.write("**Solution:** Implement transaction amount limits and real-time transaction monitoring to flag outliers.")

    elif selected_plot == "Fraud vs. Non-Fraud Transactions by Gender":
        fraud_gender = data.groupby('gender')['is_fraud'].mean().reset_index()
        fig = px.pie(fraud_gender, names='gender', values='is_fraud', title="Fraud Percentage by Gender")
        st.plotly_chart(fig)
        st.write("This pie chart shows the fraud percentage among different genders.")
        st.write("**Solution:** Apply gender-based risk profiling and enhance fraud detection algorithms for high-risk groups.")

    elif selected_plot == "Fraud Distribution by Transaction Category":
        fraud_category = data.groupby('category')['is_fraud'].mean().reset_index()
        fig = px.bar(fraud_category, x='category', y='is_fraud', title="Fraud Distribution by Transaction Category", labels={"is_fraud": "Fraud Percentage"})
        st.plotly_chart(fig)
        st.write("This graph displays the fraud percentage for each transaction category.")
        st.write("**Solution:** Implement category-specific fraud detection rules and flag unusual transactions.")

    elif selected_plot == "Fraud Percentage by Age Group":
        data['age_group'] = pd.cut(data['age'], bins=[18, 25, 35, 45, 55, 65, 100], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
        fraud_age = data.groupby('age_group')['is_fraud'].mean().reset_index()
        fig = px.bar(fraud_age, x='age_group', y='is_fraud', title="Fraud Percentage by Age Group", labels={"is_fraud": "Fraud Percentage"})
        st.plotly_chart(fig)
        st.write("This graph shows fraud percentages across different age groups.")
        st.write("**Solution:** Implement age-specific fraud prevention strategies, especially for high-risk age groups.")


    elif selected_plot == "Fraud Percentage by Hour of Day":
        fraud_hour = data.groupby('hour')['is_fraud'].mean().reset_index()
        fig = px.line(fraud_hour, x='hour', y='is_fraud', title="Fraud Percentage by Hour of Day", labels={"is_fraud": "Fraud Percentage"})
        st.plotly_chart(fig)
        st.write("This graph displays fraud percentages across different hours of the day.")
        st.write("**Solution:** Strengthen fraud detection measures during high-risk hours by implementing real-time alerts.")


    elif selected_plot == "Fraud Percentage by Day of Week":
        fraud_day = data.groupby('day')['is_fraud'].mean().reset_index()
        fraud_day['day'] = fraud_day['day'].map({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
        fig = px.line(fraud_day, x='day', y='is_fraud', title="Fraud Percentage by Day of Week")
        st.plotly_chart(fig)
        st.write("Fraud rates tend to spike on specific days.")
        st.write("**Solution:** Increase fraud monitoring and security measures on high-risk days.")


    elif selected_plot == "Fraud Percentage by Month":
        fraud_month = data.groupby('month')['is_fraud'].mean().reset_index()
        fraud_month['month'] = fraud_month['month'].map({1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"})
        fig = px.bar(fraud_month, x='month', y='is_fraud', title="Fraud Percentage by Month", color='is_fraud', color_continuous_scale="Viridis")
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(template='plotly_dark', xaxis_title="Month", yaxis_title="Fraud Percentage", font=dict(size=14))
        st.plotly_chart(fig)
        st.write("Fraud percentage varies by month, with potential spikes in holiday seasons.")
        st.write("**Solution:** Strengthen fraud detection during seasonal spikes and holiday periods.")



    elif selected_plot == "Fraudulent Transactions by State":
        if 'state' in data.columns:
            fraud_by_state = data.groupby('state')['is_fraud'].sum().reset_index()
            fig = px.scatter_geo(fraud_by_state, locations='state', locationmode="USA-states", color='is_fraud', title="Fraudulent Transactions by State", scope="usa")
            st.plotly_chart(fig)
            st.write("Fraud cases vary across states, with certain states showing higher risk levels.")
            st.write("**Solution:** Implement region-based fraud risk analysis and enhance security in high-risk states.")
        else:
            st.error("State information is missing from the dataset.")
