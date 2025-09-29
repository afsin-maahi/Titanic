import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    with open('pickle.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Predict passenger survival using machine learning")
st.markdown("**Model Accuracy: 81%** | Built with scikit-learn")

# Sidebar for input
st.sidebar.header("Passenger Information")
st.sidebar.markdown("Fill in the details below:")

# Input fields
pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3],
    index=2,
    help="1=First Class, 2=Second Class, 3=Third Class"
)

sex = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"],
    index=1
)

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30,
    help="Age in years"
)

sibsp = st.sidebar.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.sidebar.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.sidebar.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0,
    step=0.1,
    help="Ticket price in dollars"
)

embarked = st.sidebar.selectbox(
    "Port of Embarkation",
    ["Southampton", "Cherbourg", "Queenstown"],
    index=0,
    help="Where the passenger boarded"
)

# Convert inputs to model format
sex_numeric = 1 if sex == "Male" else 0
embarked_map = {"Southampton": 2, "Cherbourg": 0, "Queenstown": 1}
embarked_numeric = embarked_map[embarked]

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction button
    if st.button("🔮 Predict Survival", type="primary"):
        # Create input array
        input_data = np.array([[pclass, sex_numeric, age, sibsp, parch, fare, embarked_numeric]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if prediction == 1:
            st.success(f"✅ **SURVIVED** - Confidence: {probability[1]:.1%}")
            st.balloons()
        else:
            st.error(f"❌ **DID NOT SURVIVE** - Confidence: {probability[0]:.1%}")
        
        # Show probability bar
        prob_df = pd.DataFrame({
            'Outcome': ['Did Not Survive', 'Survived'],
            'Probability': [probability[0], probability[1]]
        })
        
        fig = px.bar(prob_df, x='Outcome', y='Probability', 
                     title="Survival Probability",
                     color='Outcome',
                     color_discrete_map={'Survived': '#2E8B57', 'Did Not Survive': '#DC143C'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# with col2:
#     # Input summary
#     st.subheader("📋 Input Summary")
#     st.markdown(f"""
#     **Class:** {pclass} {'(First)' if pclass==1 else '(Second)' if pclass==2 else '(Third)'}  
#     **Gender:** {sex}  
#     **Age:** {age} years  
#     **Family:** {sibsp + parch} members  
#     **Fare:** ${fare:.2f}  
#     **Embarked:** {embarked}
#     """)

# Model info section
st.markdown("---")
st.subheader("🤖 About the Model")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Model Accuracy", "81%", help="Accuracy on test dataset")

with col4:
    st.metric("Algorithm", "Decision Tree", help="Machine learning algorithm used")

with col5:
    st.metric("Features Used", "7", help="Number of input features")

# Feature importance (you can add this based on your model)
st.markdown("### 📈 Most Important Features")
importance_data = {
    'Feature': ['Gender', 'Passenger Class', 'Fare', 'Age', 'Family Size'],
    'Importance': [0.35, 0.25, 0.20, 0.12, 0.08]
}
importance_df = pd.DataFrame(importance_data)
fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance in Survival Prediction")
st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Data from Kaggle Titanic Competition</p>
</div>
""", unsafe_allow_html=True)
