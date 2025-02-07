🏡 Chennai House Price Prediction

This project predicts house prices in Chennai based on various features like area, number of rooms, distance from the main road, and more. 
The model is trained using Random Forest Regressor and serves predictions via a Flask web application.

---

📌 Project Overview
- Machine Learning Model: Random Forest Regressor
- Dataset: Chennai house price dataset
- Backend: Flask
- Frontend: HTML/CSS
- Model Output: Predicts house prices based on user input

---

📁 Project Structure

house_price_prediction/

│── model.py           # Train the machine learning model

│── app.py             # Flask backend to serve predictions

│── templates/

│   ├── index.html     # Webpage UI

│── static/            # (Optional) CSS/JS files for styling

│── model.pkl          # Saved trained model

│── encoder.pkl        # Saved OneHotEncoder for categorical variables

│── scaler.pkl         # Saved StandardScaler for numerical data

│── chennai_data.csv   # House price dataset

│── requirements.txt   # List of Python dependencies

│── README.md          # Project documentation

---

🚀 Setup Instructions

1️⃣ Clone the Repository
git clone https://github.com/your-username/Chennai-House-Price-Prediction.git
cd Chennai-House-Price-Prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
Run model.py to train the model and generate 3 required files:

python model.py
This will create:

model.pkl (Trained model)
encoder.pkl (For categorical feature encoding)
scaler.pkl (For feature scaling)

4️⃣ Run the Flask Application
python app.py
Open http://127.0.0.1:5000/ in your browser.
Enter details in the form to predict the house price
