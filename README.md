# American Express AI Fraud Detection System

A premium, simplified fraud detection interface built on Flask with an American Express inspired design. The app uses a pre-trained model (credit.pkl) to score transactions and returns a clear fraud risk prediction with confidence.

## Features

- American Express inspired UI with Tailwind CSS
- Simplified 8-field input form
- One-click legitimate and fraud samples
- Confidence and fraud probability display
- Loading state, secure badge, and responsive layout

## Project Structure

- app.py - Flask routes and prediction logic
- templates/index.html - Main UI and result display
- static/css/style.css - Custom premium styling
- credit.pkl - Pre-trained model

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python app.py
   ```

Open http://localhost:5000 in your browser.

## Notes

- The model expects all 30 features. The UI captures 8 and fills the rest with zeros.
- Update app.secret_key before deploying to production.
- The raw dataset files (creditcard.csv) are not included in this repo due to GitHub size limits. Download them separately and keep them local only. Link to download and access the dataset: https://www.kaggle.com/code/aarthiramalingam/creditcard
