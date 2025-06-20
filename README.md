# 🏡 House Price Prediction Web App

A Machine Learning-based interactive web application built using **Streamlit** that predicts house prices in India based on property features such as location, area, BHK configuration, construction status, RERA approval, and more.

---

### 📌 Demo Preview

> _Coming soon: Add screenshots or a GIF of your app here_

---

## 📂 Dataset Overview

The model is trained on a dataset with the following key features:

| Column Name         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `POSTED_BY`         | Who listed the property: Owner / Dealer / Builder                           |
| `UNDER_CONSTRUCTION`| Whether the property is under construction (1 = Yes, 0 = No)                |
| `RERA`              | Whether the property is RERA approved                                       |
| `BHK_NO.`           | Number of bedrooms                                                          |
| `BHK_OR_RK`         | Type: "BHK" (Bedroom Hall Kitchen) or "RK" (Room Kitchen)                   |
| `SQUARE_FT`         | Area of the house in square feet                                            |
| `READY_TO_MOVE`     | Whether the property is ready to move (1 = Yes, 0 = No)                     |
| `RESALE`            | Whether the property is being resold                                        |
| `ADDRESS`           | Location of the property                                                    |
| `LONGITUDE`         | Longitude coordinate of the property                                        |
| `LATITUDE`          | Latitude coordinate of the property                                         |
| `TARGET(PRICE_IN_LACS)` | Final property price in Lakhs (₹) - **Target variable**                  |

---

## 🚀 Features

- 🔍 **Train ML models** (Linear Regression, Decision Tree, Random Forest, SVM, XGBoost)
- 📊 **Evaluate performance** (R² Score, RMSE)
- 📈 **Predict prices** on new CSV data
- 💾 **Download predictions** as a CSV file
- ✅ Clean & interactive **Streamlit interface**

---

## 🔧 How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
2. Install Requirements
We recommend creating a virtual environment first:

bash
Copy
Edit
pip install -r requirements.txt
If you don’t have a requirements.txt, here are the core libraries you need:

bash
Copy
Edit
pip install streamlit pandas scikit-learn xgboost matplotlib tensorflow
3. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
🖥️ App Usage Instructions
🔹 Training a Model
Upload your training CSV file containing property data (with price).

Choose your ML model from the dropdown (e.g., XGBoost).

Click "Train Model" to train and evaluate.

🔹 Predicting on New Data
Upload a test dataset CSV (same format as training, but without price).

The app will preprocess the data and show predictions.

Click "Download Predictions" to save as CSV.

📁 Project Structure
bash
Copy
Edit
house-price-prediction/
├── app.py                # Main Streamlit app
├── tools.py              # All training, preprocessing, scoring functions
├── train.csv             # (Optional) Sample training dataset
├── test.csv              # (Optional) Sample test dataset
├── requirements.txt      # Dependencies
└── README.md             # This file
⚙️ Internals – How It Works
Data Preprocessing: Label encoding, one-hot encoding (POSTED_BY), and feature dropping (ADDRESS, LATITUDE, LONGITUDE)

Model Training: Supports multiple regression models

Evaluation: Calculates R² and RMSE on a validation split (80/20)

Predictions: Applies trained model on new data with the same preprocessing pipeline

📌 Notes
This app works with CSV files only.

Make sure test data format matches the training data (excluding the target column).

Neural Networks & Polynomial regression are excluded for simplicity & speed.

💡 Future Enhancements
📍 Map-based predictions using latitude and longitude

📊 Visualizations: Actual vs Predicted

🌐 Cloud deployment via Streamlit Cloud or Hugging Face Spaces

🧠 Add ensemble stacking for model fusion

🙌 Contributing
Pull requests are welcome! For major changes, please open an issue first.

📜 License
This project is open source under the MIT License.

✉️ Contact
Developed by Nimish Bagwale
📧 Contact: add-your-email
🌐 Portfolio: add-your-portfolio-link

yaml
Copy
Edit

---

Let me know if you’d like the following:
- A ready-made `requirements.txt`
- A sample `train.csv` and `test.csv`
- GitHub Actions for auto-deploy on push

I'm here to supercharge your repo 🚀
