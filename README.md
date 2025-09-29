🚢 Titanic Survival Prediction Web App
A machine learning web application that predicts passenger survival on the Titanic using historical data. Built with Python and Streamlit for an interactive, user-friendly experience.

📊 Project Overview
This project implements a machine learning model to predict whether a passenger would survive the Titanic disaster based on features like gender, age, passenger class, and fare. The model achieves 81% accuracy and is deployed through an intuitive Streamlit web interface where users can input passenger details to get real-time predictions.

Key Features
Interactive Streamlit Interface: Beautiful, responsive web app with real-time predictions
Visual Results: Probability charts and feature importance graph
81% Model Accuracy: Trained on historical Titanic dataset
One-Click Deployment: Easy setup and deployment
Professional Dashboard: Clean, modern UI perfect for portfolios

🛠️ Tech Stack
Machine Learning: Python, Scikit-learn, Pandas, NumPy
Web Framework: Streamlit
Data Visualization: Plotly
Model Serialization: Pickle
Development: Jupyter Notebooks for model training

📈 Model Performance
Algorithm: Decision Trree ( depth = 7 )
Training Accuracy: 81%
Dataset: Titanic passenger data from Kaggle
Key Features: Gender, Passenger Class, Age, Fare, Family Size, Embarkation Port

Most Important Features:
Gender - Primary survival predictor (women had higher survival rates)
Passenger Class - First-class passengers had better survival chances
Fare - Higher fares correlated with better cabin locations
Age - Children were prioritized during evacuation

🚀 Getting Started
Prerequisites
Python 3.7+

Installation
Clone the repository
bash
git clone https://github.com/yourusername/titanic-prediction-app.git
cd titanic-prediction-app
Install dependencies
bash
pip install -r requirements.txt
Run the application
bash
streamlit run app.py
Open your browser
The app will automatically open at http://localhost:8501
If not, manually navigate to the URL shown in terminal

💻 Usage
Input passenger details in the sidebar:
Passenger Class (1st, 2nd, or 3rd)
Gender (Male/Female)
Age in years
Number of siblings/spouses aboard
Number of parents/children aboard
Ticket fare paid
Port of embarkation
Click "Predict Survival" to get results

View prediction results:
Survival prediction with confidence percentage
Interactive probability chart
Feature importance visualization

🔬 Model Details
The model was trained on the famous Titanic dataset with the following approach:
Data Preprocessing: Handled missing values, encoded categorical variables , Scaling features
Model Training:Decision Tree ( depth = 7 )
Evaluation: Achieved 81% accuracy on test set

🛠️ Technical Requirements
requirements.txt:
streamlit==1.28.1
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
plotly==5.17.0

🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a Pull Request

🙏 Acknowledgments
Dataset source: Kaggle Titanic Competition
Built with Streamlit - The fastest way to build data apps
Visualization powered by Plotly

⭐ Star this repository if you found it helpful! ⭐
