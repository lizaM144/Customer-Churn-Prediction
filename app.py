# # Gender -> 1 Female 0 Male
# # Churn -> 1 Yes 0 No
# # InternetService -> One-Hot Encoding
# # TechSupport -> 1 Yes 0 No

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')
features = joblib.load('feature_names.pkl') # Load feature names

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("Enter customer details below to calculate the risk of churn.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=130, value=12)

with col2:
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=20.0, max_value=200.0, value=70.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No Internet"])
    tech_support = st.selectbox("Has Tech Support?", ["Yes", "No"])


predict_button = st.button("Predict Churn")

if predict_button:
    
    # mapping the categorical inputs to numbers
    gender_val = 1 if gender == "Female" else 0
    
    # we only check for Fiber and No Internet since DSL is the base case
    is_fiber = 1 if internet_service == "Fiber Optic" else 0
    is_no_internet = 1 if internet_service == "No Internet" else 0
    
    # Tech Support Mapping
    tech_yes = 1 if tech_support == "Yes" else 0
    
    # array with all features in order
    input_dict = {
    'Age': age,
    'Gender': gender_val,
    'Tenure': tenure,
    'MonthlyCharges': monthly_charge,
    'InternetService_Fiber Optic': is_fiber,
    'InternetService_No Internet': is_no_internet,
    'TechSupport_Yes': tech_yes
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[features]  # enforce order
    
    # scale only numeric columns
    input_df[['Age', 'Tenure', 'MonthlyCharges']] = scaler.transform(input_df[['Age', 'Tenure', 'MonthlyCharges']])
    input_data = input_df

    st.divider()
    # prediction
    probability = model.predict_proba(input_data)[0][1] # Probability of Churn (Class 1)
    
    # results
    st.subheader("Prediction Result")
    
    prob_percentage = probability * 100
    
    if prob_percentage < 30:
        st.success(f"Low Risk: {prob_percentage:.1f}% Probability of Churn")
        st.write("Suggestion: Customer is safe. Keep engaging them with standard offers.")
    elif prob_percentage < 70:
        st.warning(f"Medium Risk: {prob_percentage:.1f}% Probability of Churn")
        st.write("Suggestion: Customer is at risk. Consider offering a small discount or free service upgrade.")
    else:
        st.error(f"High Risk: {prob_percentage:.1f}% Probability of Churn")
        st.write("Suggestion: HIGH CHURN RISK! Immediate action required. Offer a long-term contract discount or dedicated support.")

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    
    # --- 6. SHAP Explanation ---
    st.divider()
    st.subheader("Why this result?")
    st.write("The chart below shows which features pushed the risk UP (Red) or DOWN (Blue).")
    
    # Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)
    st.pyplot(plt.gcf())

###
    # Smart Explanation
    st.divider()
    st.subheader("Key Factors Behind this Result")
    
    # Extract the values for the single prediction (Row 0, Class 1)
    # .values returns the raw numbers
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Value': input_data.iloc[0].values, # The actual input values (0 or 1)
        'Contribution': shap_values[0, :, 1].values
    })
    
    # Separate into "Risk Factors" (Positive) and "Safety Factors" (Negative)
    risk_factors = feature_importance[feature_importance['Contribution'] > 0.02].sort_values(by='Contribution', ascending=False)
    safety_factors = feature_importance[feature_importance['Contribution'] < 0.02].sort_values(by='Contribution', ascending=True)
    

    def key_driver_text(row):
        feature = row['Feature']
        value = row['Value']

        # For Tech support part
        if feature == 'TechSupport_Yes':
            return "Has Tech Support" if value == 1 else "No Tech Support"
        # for Internet Service part - Fiber
        if feature == 'InternetService_Fiber Optic':
            return "Uses Fiber Optic Internet" if value == 1 else "Does not use Fiber Optic Internet"
        # for Internet Service part - No Internet
        if feature == 'InternetService_No Internet':
            return "No Internet Service" if value == 1 else "Has Internet Service (DSL/Fiber)"
        # for gender
        if feature == 'Gender':
            return "Female" if value == 1 else "Male"
        
        # the numerical features like (Age, Tenure, MonthlyCharges) will stay the same
        if feature == 'Age':
            return f"Age: {age} years"
        if feature == 'Tenure':
            return f"Tenure: {tenure} months"
        if feature == 'MonthlyCharges':
            return f"Monthly Charges: ${monthly_charge}"
        return feature
    
    col_risk, col_safe = st.columns(2)
    
    with col_risk:
        st.markdown("**Risk Drivers**")
        st.caption("These factors are PUSHING the customer to leave.")
        if not risk_factors.empty:
            for index, row in risk_factors.head(3).iterrows():
                friendly_text = key_driver_text(row)
                st.error(f"**{friendly_text}**")
        else:
            st.write("No major risk factors found!")

    with col_safe:
        st.markdown("**Safety Drivers**")
        st.caption("These factors are KEEPING the customer loyal.")
        if not safety_factors.empty:
            for index, row in safety_factors.head(3).iterrows():
                friendly_text = key_driver_text(row)
                st.success(f"**{friendly_text}**")
        else:
            st.write("No major safety factors found!")

    
    st.divider()
    st.subheader("Recommended Actions")
    st.markdown("Based on the key risk drivers identified above, here are top strategies to retain this customer:")

    # Function to get solution based on feature and value
    def get_solution(row):
        feature = row['Feature']
        value = row['Value']
        
        # No Tech Support
        if feature == "TechSupport_Yes" and value == 0:
            return "**Offer Free Tech Support:** This customer lacks support, which is a #1 churn driver. Offer a 3-month free trial of Premium Tech Support."
        
        # Fiber Optic User 
        if feature == "InternetService_Fiber Optic" and value == 1:
            return "**VIP Bundle Offer:** Fiber customers expect premium service. Check if a competitor is offering a lower price and match it, or offer a free speed upgrade."
        
        # customer with no internet could leave for competitor
        if feature == "InternetService_No Internet" and value == 1:
            return "**Digital Onboarding:** This customer has no internet. Call them to see if they are interested in a basic DSL starter package for $20/mo."

        # high or low monthly charges
        if feature == "MonthlyCharges":
            if monthly_charge > 80:
                return "**Loyalty Discount:** This customer's bill is high. Offer a 10% discount to lock them in for another year."
            
            elif monthly_charge < 30:
                return "**Service Upgrade:** This customer is on a basic low-tier plan. Offer a free upgrade to the next tier to increase value."
            
            else:
                 return "**Value Review:** Review their usage and ensure they are getting the best value for their money."

        # low tenure means new customer
        if feature == "Tenure":
            return "**Onboarding Call:** This is a new customer. Schedule a 'Happiness Check' call to ensure their setup is working perfectly."

        # gender is not a strong factor but we can add a general suggestion
        if feature == "Gender":
            return "**General Appreciation:** Send a personalized 'Thank You' email to build rapport."

        return None

    suggestions = [] # empty list to hold suggestions
    
    # Loop through the risk factors and get solutions
    if not risk_factors.empty:
        for index, row in risk_factors.head(3).iterrows():
            sol = get_solution(row)
            if sol:
                suggestions.append(sol)

    if len(suggestions) > 0:
        # showing only top 2 suggestions as per risk factors
        for s in suggestions[:2]: 
            st.info(s)
    else:
        st.success("No immediate actions needed! Keep up the good work.")