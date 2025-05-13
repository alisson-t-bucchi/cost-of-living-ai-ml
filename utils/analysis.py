def get_available_countries(df):
    return sorted(df["country"].unique())

def get_cities_by_country(df, country):
    return sorted(df[df["country"] == country]["city"].unique())

def get_city_data(df, country, city):
    return df[(df["country"] == country) & (df["city"] == city)].iloc[0]

def get_comparison_dataframe(df, country):
    compare_df = df[df["country"] == country][[
        "city", "Estimated_Monthly_Cost", "Avg_Monthly_Salary", "Cost_to_Salary_Ratio"
    ]].dropna().set_index("city")
    return compare_df
