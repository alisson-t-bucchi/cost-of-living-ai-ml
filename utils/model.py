from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_cost_model(df: pd.DataFrame):
    features = ["Meal_Inexpensive", "Monthly_Transport_Pass", "Rent_City_Center_1BR", "Basic_Utilities"]
    df = df.dropna(subset=features + ["Estimated_Monthly_Cost"])
    X = df[features]
    y = df["Estimated_Monthly_Cost"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse
