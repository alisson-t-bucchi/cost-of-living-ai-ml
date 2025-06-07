import streamlit as st
from utils.data_loader import load_and_prepare_data
from utils.analysis import (
    get_available_countries,
    get_cities_by_country,
    get_city_data,
    get_comparison_dataframe
)
from utils.visualization import plot_cost_vs_salary
from utils.model import train_cost_model

st.set_page_config(page_title="Global cost of living predictor", layout="wide")
st.title("📊 Cost of Living Analyses and Forecasts by City")

df = load_and_prepare_data("data/cost-of-living.csv")

countries = get_available_countries(df)
selected_country = st.selectbox("🌍 Select Country", countries)

cities = get_cities_by_country(df, selected_country)
selected_city = st.selectbox("🏙️ Select City", cities)

# Selected city
city_data = get_city_data(df, selected_country, selected_city)

# City selected dates
st.subheader(f"📍 Results for {selected_city} in USD:")
st.write("💰 **Estimated Monthly Cost:**", f"{city_data['Estimated_Monthly_Cost']:.2f} USD")
st.write("📈 **Montly Salary Average:**", f"{city_data['Avg_Monthly_Salary']:.2f} USD")
st.write("⚖️ **Cost to salary ratio:**", f"{city_data['Cost_to_Salary_Ratio']:.2f}")

# Previsão com IA
st.subheader(f"🤖 Cost of Living Forecasting for {selected_city}")
st.write(f"💹 Linear Regression Analysis")

model, mse = train_cost_model(df)

st.write(f"📉 Mean Square Error (MSE) of the model: **{mse:.2f} USD²**")

if st.button("🔮 Predict AI costs for this city"):
    features = [
        city_data["Meal_Inexpensive"],
        city_data["Monthly_Transport_Pass"],
        city_data["Rent_City_Center_1BR"],
        city_data["Basic_Utilities"]
    ]
    predicted_cost = model.predict([features])[0]
    st.success(f"✅ Cost predicted by the model: **{predicted_cost:.2f} USD**")

# Graphic comparing cities from a selected country.
st.subheader("📊 Comparison with other cities in the country")
compare_df = get_comparison_dataframe(df, selected_country)

if not compare_df.empty:
    st.pyplot(plot_cost_vs_salary(compare_df))
    st.subheader("📈 Cost to Salary Ratio:")
    st.write("⚠️ Values > 1.0 indicate expensive(s) city(ies)!")
    st.dataframe(compare_df.sort_values("Cost_to_Salary_Ratio")[["Cost_to_Salary_Ratio"]])
else:
    st.warning("❗ No cities with enough dates to compare!")

