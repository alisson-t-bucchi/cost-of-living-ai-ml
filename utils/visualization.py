import matplotlib.pyplot as plt

def plot_cost_vs_salary(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df[["Estimated_Monthly_Cost", "Avg_Monthly_Salary"]]\
        .sort_values("Estimated_Monthly_Cost", ascending=False)\
        .plot(kind="bar", ax=ax)

    plt.title("Estimated Cost vs Average Salary")
    plt.ylabel("USD")
    plt.tight_layout()
    return fig
