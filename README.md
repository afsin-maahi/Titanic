

 Titanic Survival Prediction Web App
A machine learning web application that predicts passenger survival on the Titanic using historical data. Built with Python and deployed as an interactive web interface.

 Project Overview
This project implements a machine learning model to predict whether a passenger would survive the Titanic disaster based on features like gender, age, passenger class, and fare. The model achieves 81% accuracy and is deployed through an intuitive web interface where users can input passenger details to get real-time predictions.

Key Features
Interactive Web Interface: User-friendly form for entering passenger details

Real-time Predictions: Instant survival probability calculations

81% Model Accuracy: Trained on historical Titanic dataset

Responsive Design: Works on desktop and mobile devices

Input Validation: Ensures data quality before prediction

 Tech Stack
Machine Learning: Python, Scikit-learn, Pandas, NumPy

Web Framework: Streamlit 

Model Serialization: Pickle

Frontend: HTML, CSS, Bootstrap (for Flask version)

Data Processing: Pandas for data manipulation

 Model Performance
Algorithm: DecisionTree

Training Accuracy: 81%

Dataset: Titanic passenger data from Kaggle

Key Features: Gender, Passenger Class, Age, Fare, Family Size, Embarkation Port

 Getting Started
Prerequisites
Python 3.7+

pip package manager

Installation
Clone the repository

bash
git clone https://github.com/yourusername/titanic-prediction-app.git
cd titanic-prediction-app
Install dependencies

bash
pip install streamlit pandas numpy scikit-learn
# OR for Flask version:
# pip install flask pandas numpy scikit-learn
Run the application

bash
# For Streamlit version:
streamlit run app.py

# For Flask version:
# python app.py
Open your browser

Streamlit: http://localhost:8501

Flask: http://localhost:5000

💻 Usage
Open the web application in your browser

Fill in the passenger details:

Passenger Class: 1st, 2nd, or 3rd class

Gender: Male or Female

Age: Passenger's age in years

Family Members: Number of siblings/spouses and parents/children aboard

Fare: Ticket price paid

Embarkation Port: Where the passenger boarded (Southampton, Cherbourg, Queenstown)

Click "Predict Survival" to get the result

View the prediction result with confidence percentage

📁 Project Structure
text
titanic-prediction-app/
│
├── app.py                 # Main application file
├── pickle.pkl            # Trained ML model
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
│
├── templates/           # HTML templates (Flask version)
│   └── index.html
│
├── data/               # Dataset files
│   └── titanic.csv
│
└── notebooks/          # Jupyter notebooks
    └── model_training.ipynb
 Model Details
The model was trained on the famous Titanic dataset with the following approach:

Data Preprocessing: Handled missing values, encoded categorical variables, Scaling the values

Model Training: Decision tree ( max_depth=7)

Evaluation: Achieved 81% accuracy on test set

Most Important Features:
Gender - Primary survival predictor (women had higher survival rates)

Passenger Class - First-class passengers had better survival chances

Fare - Higher fares correlated with better cabin locations

Age - Children were prioritized during evacuation

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -am 'Add new feature')

Push to the branch (git push origin feature/new-feature)

Create a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

 Acknowledgments
Dataset source: Kaggle Titanic Competition

Inspiration from the machine learning community

Thanks to contributors and reviewers

Project Link: https://github.com/yourusername/titanic-prediction-app

⭐ Star this repository if you found it helpful! ⭐

