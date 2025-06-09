# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df2 = pd.read_csv("Unemployment in India.csv")

# Add source column to each
df1["Source"] = "Dataset 1"
df2["Source"] = "Dataset 2"

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Select common columns
common_columns = [
    "Region", "Date", "Estimated Unemployment Rate (%)",
    "Estimated Employed", "Estimated Labour Participation Rate (%)", "Source"
]
df1_clean = df1[common_columns]
df2_clean = df2[common_columns]

# Merge datasets
df = pd.concat([df1_clean, df2_clean], ignore_index=True)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Drop rows with invalid/missing dates
df = df.dropna(subset=['Date'])

# Sort by date
df.sort_values(by='Date', inplace=True)

# Add Month-Year for time grouping
df['Month-Year'] = df['Date'].dt.to_period('M').astype(str)

# ----- Visualizations -----

# Set seaborn style
sns.set(style="whitegrid")

# 1. Unemployment Rate Over Time by Region
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', legend=False)
plt.title('Unemployment Rate Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Average Unemployment Rate by Region
plt.figure(figsize=(12, 6))
region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
sns.barplot(x=region_avg.index, y=region_avg.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Unemployment Rate by Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.tight_layout()
plt.show()

# 3. Heatmap of Unemployment Rate by Region and Month
pivot_table = df.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Region',
    columns='Month-Year',
    aggfunc='mean'
)

plt.figure(figsize=(16, 8))
sns.heatmap(pivot_table, cmap='coolwarm', linecolor='white', linewidths=0.1)
plt.title('Unemployment Rate Heatmap by Region and Month')
plt.xlabel('Month-Year')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

# 4. Summary statistics
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())
