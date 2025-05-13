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

# Configurações iniciais
st.set_page_config(page_title="Custo de Vida + IA", layout="wide")
st.title("📊 Análise e Previsão de Custo de Vida por Cidade")

# Carregamento e preparação dos dados
df = load_and_prepare_data("data/cost-of-living.csv")

# Filtros de país e cidade
countries = get_available_countries(df)
selected_country = st.selectbox("🌍 Selecione um país", countries)

cities = get_cities_by_country(df, selected_country)
selected_city = st.selectbox("🏙️ Selecione uma cidade", cities)

# Dados da cidade selecionada
city_data = get_city_data(df, selected_country, selected_city)

# Exibição dos dados
st.subheader(f"📍 Resultados para {selected_city}")
st.write("💰 **Custo mensal estimado:**", f"{city_data['Estimated_Monthly_Cost']:.2f} USD")
st.write("📈 **Salário líquido médio:**", f"{city_data['Avg_Monthly_Salary']:.2f} USD")
st.write("⚖️ **Proporção Custo/Salário:**", f"{city_data['Cost_to_Salary_Ratio']:.2f}")

# Previsão com IA
st.subheader("🤖 Previsão de Custo de Vida com Regressão Linear")

model, mse = train_cost_model(df)

st.write(f"📉 Erro Quadrático Médio (MSE) do modelo: **{mse:.2f} USD²**")

if st.button("🔮 Prever custo com IA para esta cidade"):
    features = [
        city_data["Meal_Inexpensive"],
        city_data["Monthly_Transport_Pass"],
        city_data["Rent_City_Center_1BR"],
        city_data["Basic_Utilities"]
    ]
    predicted_cost = model.predict([features])[0]
    st.success(f"✅ Custo previsto pelo modelo: **{predicted_cost:.2f} USD**")

# Gráfico comparativo entre cidades do país
st.subheader("📊 Comparação com outras cidades do país")
compare_df = get_comparison_dataframe(df, selected_country)

if not compare_df.empty:
    st.pyplot(plot_cost_vs_salary(compare_df))
    st.dataframe(compare_df.sort_values("Cost_to_Salary_Ratio")[["Cost_to_Salary_Ratio"]])
else:
    st.warning("❗ Nenhuma cidade com dados suficientes para comparação.")

