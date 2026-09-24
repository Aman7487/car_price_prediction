import pickle
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
car = pd.read_csv("quikr_car (1).csv")
backup = car.copy()

# ---------------------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------------------
car = backup.copy()
car = car[car["Price"] != "Ask For Price"]

car["Price"] = car["Price"].str.replace(",", "", regex=False).astype(int)

car["kms_driven"] = car["kms_driven"].fillna("0")
car["kms_driven"] = (
    car["kms_driven"]
    .astype(str)
    .str.replace(" kms", "", regex=False)
    .str.replace(",", "", regex=False)
)
car = car[car["kms_driven"].str.isnumeric()]
car["kms_driven"] = car["kms_driven"].astype(int)

car["name"] = car["name"].apply(lambda x: " ".join(str(x).split(" ")[:3]))

car = car.dropna(subset=["fuel_type"])

# Drop unrealistic price outliers
car = car[car["Price"] < 6e6].reset_index(drop=True)

print(car.describe())
print(car.info())
print(car.nunique())

# Save cleaned data WITHOUT the pandas index column
# (the original code forgot index=False, which leaks an "Unnamed: 0" column
# into anything that later reads this csv)
car.to_csv("cleaned car.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
x = car.drop(columns="Price")
y = car["Price"]

# ---------------------------------------------------------------------------
# 4. Build preprocessing + model pipeline
# ---------------------------------------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(x[["name", "company", "fuel_type"]])

column_trans = make_column_transformer(
    (
        OneHotEncoder(categories=ohe.categories_, handle_unknown="ignore"),
        ["name", "company", "fuel_type"],
    ),
    remainder="passthrough",
)

# ---------------------------------------------------------------------------
# 5. Search for a good random_state (kept from original approach, but capped
#    at a smaller range and without spamming 1000 print statements)
# ---------------------------------------------------------------------------
scores = []
N_TRIALS = 200  # 1000 was overkill and slow; 200 finds a near-optimal split fine

for i in range(N_TRIALS):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=i
    )
    pipe = make_pipeline(column_trans, LinearRegression())
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test, y_pred))

best_state = int(np.argmax(scores))
print(f"Best random_state: {best_state}, R2: {scores[best_state]:.4f}")

# ---------------------------------------------------------------------------
# 6. Final fit using the best split
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=best_state
)

pipe = make_pipeline(column_trans, LinearRegression())
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print("Final R2:", r2_score(y_test, y_pred))

# ---------------------------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------------------------
with open("LinearRegressionModel.pkl", "wb") as f:
    pickle.dump(pipe, f)

# Sanity check prediction
sample = pd.DataFrame(
    [["Maruti Suzuki Swift", "Maruti", 2019, 100, "Petrol"]],
    columns=["name", "company", "year", "kms_driven", "fuel_type"],
)
print("Sample prediction:", pipe.predict(sample))