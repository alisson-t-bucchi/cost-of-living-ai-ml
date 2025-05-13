import pandas as pd

csv_path = "data/cost-of-living.csv"

def load_and_prepare_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"])

    df.rename(columns={
        "x1": "Meal_Inexpensive",
        "x29": "Monthly_Transport_Pass",
        "x36": "Basic_Utilities",
        "x48": "Rent_City_Center_1BR",
        "x54": "Avg_Monthly_Salary",
        "x55": "Mortgage_Interest_Rate"
    }, inplace=True)

    df = df[df["data_quality"] == 1]

    df["Estimated_Monthly_Cost"] = (
        df["Meal_Inexpensive"] * 30 +
        df["Monthly_Transport_Pass"] +
        df["Basic_Utilities"] +
        df["Rent_City_Center_1BR"]
    )

    df["Cost_to_Salary_Ratio"] = df["Estimated_Monthly_Cost"] / df["Avg_Monthly_Salary"]
    return df

df = load_and_prepare_data(csv_path)



