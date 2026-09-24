# 🚗 Car Price Prediction

A machine learning project that predicts used car prices based on the car's name, company, manufacturing year, kilometers driven, and fuel type. Built with **scikit-learn** (Linear Regression) and served via both a **Streamlit** web app and a **FastAPI** REST API.

## Demo

- **Streamlit app:** `streamlit run app.py`
- **FastAPI docs:** `uvicorn main:app --reload` → visit `http://127.0.0.1:8000/docs`

## Project Structure

```
.
├── train_model.py           # Cleans data, trains the model, saves the .pkl
├── app.py                   # Streamlit front-end
├── main.py                  # FastAPI backend
├── requirements.txt         # Python dependencies
├── quikr_car (1).csv        # Raw dataset (input to train_model.py)
├── cleaned car.csv          # Cleaned dataset (output of train_model.py)
└── LinearRegressionModel.pkl # Trained model (output of train_model.py)
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the model (only needed if you don't already have the `.pkl` and cleaned csv)
```bash
python train_model.py
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. Run the FastAPI backend
```bash
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs` for interactive API docs.

## Tech Stack

- Python, pandas, numpy
- scikit-learn (OneHotEncoder + LinearRegression pipeline)
- Streamlit
- FastAPI + uvicorn

## Dataset

Source: [Quikr car dataset] — includes name, company, year, price, kms driven, and fuel type for used cars.

## License

MIT (or your choice)
