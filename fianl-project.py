import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load and preprocess data
df = pd.read_csv("human-development-indicators-for-guyana-2.csv")
columns_of_interest = ['#date+year', '#indicator+code', '#indicator+value+num']
df_subset = df[columns_of_interest]
df_subset = df_subset.drop_duplicates(subset=['#date+year', '#indicator+code'], keep='last')
df_subset['#date+year'] = df_subset['#date+year'].apply(lambda x: x.split('/')[0] if '/' in str(x) else x)
df_subset['#date+year'] = pd.to_numeric(df_subset['#date+year'], errors='coerce')
df_pivot = df_subset.pivot_table(index='#date+year', columns='#indicator+code', values='#indicator+value+num', aggfunc='first')
df_pivot.reset_index(inplace=True)

def save_and_show(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 1. Line graph: HDI over time (for index.html)
plt.figure(figsize=(12, 6))
plt.plot(df_pivot['#date+year'], df_pivot[137506], marker='o')
plt.title("Human Development Index (HDI) for Guyana Over Time")
plt.xlabel("Year")
plt.ylabel("HDI")
plt.grid(True)
save_and_show('hdi_over_time.png')

# 2. Stacked area chart: Components of HDI over time (for index.html)
plt.figure(figsize=(12, 6))
plt.stackplot(df_pivot['#date+year'],
              df_pivot[103206], df_pivot[103606], df_pivot[103706],
              labels=['Life Expectancy Index', 'Income Index', 'Education Index'])
plt.title("Components of HDI for Guyana Over Time")
plt.xlabel("Year")
plt.ylabel("Index Value")
plt.legend(loc='upper left')
plt.grid(True)
save_and_show('hdi_components_area.png')

# 3. Bar chart: HDI comparison for selected years (for index.html)
selected_years = [1990, 2000, 2010, 2019]
hdi_values = [df_pivot[df_pivot['#date+year'] == year][137506].values[0] for year in selected_years]
plt.figure(figsize=(10, 6))
plt.bar(selected_years, hdi_values)
plt.title("HDI Comparison for Selected Years")
plt.xlabel("Year")
plt.ylabel("HDI")
for i, v in enumerate(hdi_values):
    plt.text(selected_years[i], v, f'{v:.3f}', ha='center', va='bottom')
save_and_show('hdi_comparison_bar.png')

# 4. Scatter plot: Life Expectancy vs GNI per capita (for analysis.html)
plt.figure(figsize=(10, 6))
plt.scatter(df_pivot[195706], df_pivot[69206])
plt.title("Life Expectancy vs GNI per capita")
plt.xlabel("GNI per capita (constant 2017 PPP$)")
plt.ylabel("Life Expectancy (years)")
plt.grid(True)
save_and_show('life_expectancy_vs_gni_scatter.png')

# 5. Grouped bar chart: Education Indicators (for analysis.html)
years = [1990, 2000, 2010, 2019]
mean_schooling = [df_pivot[df_pivot['#date+year'] == year][103006].values[0] for year in years]
expected_schooling = [df_pivot[df_pivot['#date+year'] == year][69706].values[0] for year in years]

x = np.arange(len(years))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, mean_schooling, width, label='Mean Years of Schooling')
rects2 = ax.bar(x + width/2, expected_schooling, width, label='Expected Years of Schooling')

ax.set_ylabel('Years')
ax.set_xlabel('Year')
ax.set_title('Education Indicators Over Time')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()

fig.tight_layout()
save_and_show('education_indicators_bar.png')

# 6. Heatmap: Correlation of key indicators (for analysis.html)
key_indicators = [69206, 103006, 195706, 137506]  # Life expectancy, Mean schooling, GNI, HDI
correlation_matrix = df_pivot[key_indicators].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Key Development Indicators")
save_and_show('key_indicators_heatmap.png')

# 7. Line graph: GNI per capita growth (for conclusion.html)
plt.figure(figsize=(12, 6))
plt.plot(df_pivot['#date+year'], df_pivot[195706], marker='o')
plt.title("GNI per capita Growth for Guyana")
plt.xlabel("Year")
plt.ylabel("GNI per capita (constant 2017 PPP$)")
plt.grid(True)
save_and_show('gni_per_capita_growth.png')

# 8. Radar chart: HDI components for latest year (for conclusion.html)
latest_year = df_pivot['#date+year'].max()
latest_data = df_pivot[df_pivot['#date+year'] == latest_year]

categories = ['Life Expectancy Index', 'Education Index', 'Income Index']
values = [
    latest_data[103206].values[0],
    latest_data[103706].values[0],
    latest_data[103606].values[0]
]

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
values = np.concatenate((values, [values[0]]))  # repeat the first value to close the polygon
angles = np.concatenate((angles, [angles[0]]))  # repeat the first angle to close the polygon

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
ax.set_title(f"HDI Components for Guyana ({latest_year})")
ax.grid(True)
save_and_show('hdi_components_radar.png')

# Regression analysis (optional, for additional insights)
X = df_pivot[[195706, 69206, 103006]].dropna()  # GNI per capita, Life expectancy, Mean years of schooling
y = df_pivot[137506].loc[X.index]  # HDI

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")