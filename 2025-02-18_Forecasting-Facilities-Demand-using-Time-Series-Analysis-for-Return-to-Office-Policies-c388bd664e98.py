# Description: Short example for Forecasting Facilities Demand using Time Series Analysis for Return to Office Policies.



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
import signalplot
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)




# Set parameters for simulation
n_employees = 1000
n_days = 300
# Generate badge-in and badge-out times with variability
badge_data = []
for emp_id in range(1, n_employees + 1):
    for day in range(1, n_days + 1):
        if np.random.rand() < 0.6:  # 60% chance the employee comes to the office
            badge_in_time = np.random.normal(loc=9, scale=1)  # Around 9 AM
            badge_out_time = badge_in_time + np.random.normal(loc=8, scale=1.5)  # 8-hour shift
            badge_data.append([emp_id, day, max(7, min(11, badge_in_time)), max(15, min(20, badge_out_time))])
# Convert to DataFrame
df_badge = pd.DataFrame(badge_data, columns=["Employee_ID", "Day", "Badge_In", "Badge_Out"])
# Aggregate badge data by day
daily_badges = df_badge.groupby("Day").size()
# Plot daily badge-in counts
plt.figure(figsize=(12, 6))
plt.plot(daily_badges.index, daily_badges.values, color="black", label="Employees Badged In")
plt.xlabel("Day")
plt.ylabel("Number of Employees in Office")
plt.title("Daily Employee Badge-Ins Over 300 Days")
plt.legend()
# Save and show
plt.savefig("badge_data_time_series.png")
plt.show()


# Fit an ARIMA model to forecast attendance
arima_model = ARIMA(daily_badges, order=(2,1,2)).fit()
forecast = arima_model.forecast(30)  # Predict next 30 days
# Plot actual data and forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_badges.index, daily_badges.values, label="Actual Attendance", color="black")
plt.plot(range(n_days, n_days+30), forecast, label="Forecasted Attendance", color="red", linestyle="dashed")
plt.xlabel("Day")
plt.ylabel("Number of Employees in Office")
plt.title("ARIMA Forecast for Future Badge-Ins")
plt.legend()
# Save and show
plt.savefig("badge_in_forecast.png")
plt.show()


# Set Tufte-like style
signalplot.apply(font_family='serif')

# -------------------------------
# Simulating "Coffee Badgers" in the Badge-In Data
# -------------------------------

# Set parameters for simulation
n_employees = 1000
n_days = 300

# Generate badge-in and badge-out times with variability
badge_data = []

for emp_id in range(1, n_employees + 1):
    for day in range(1, n_days + 1):
        if np.random.rand() < 0.6:  # 60% chance the employee comes to the office
            badge_in_time = np.random.normal(loc=9, scale=1)  # Around 9 AM
            badge_out_time = badge_in_time + np.random.normal(loc=8, scale=1.5)  # 8-hour shift
            badge_data.append([emp_id, day, max(7, min(11, badge_in_time)), max(15, min(20, badge_out_time))])

# Convert to DataFrame
df_badge = pd.DataFrame(badge_data, columns=["Employee_ID", "Day", "Badge_In", "Badge_Out"])

# Introduce "Coffee Badgers"
coffee_badger_percentage = 0.1  # 10% of employees are coffee badgers
coffee_badger_ids = np.random.choice(df_badge["Employee_ID"].unique(), 
                                     size=int(n_employees * coffee_badger_percentage), 
                                     replace=False)

# Update badge-out times for coffee badgers (leave within an hour)
df_badge.loc[df_badge["Employee_ID"].isin(coffee_badger_ids), "Badge_Out"] = df_badge.loc[
    df_badge["Employee_ID"].isin(coffee_badger_ids), "Badge_In"] + np.random.uniform(0.2, 1.0, 
    size=len(df_badge[df_badge["Employee_ID"].isin(coffee_badger_ids)]))

# Add a classification column
df_badge["Coffee_Badger"] = df_badge.apply(lambda row: 1 if row["Badge_Out"] - row["Badge_In"] < 1 else 0, axis=1)

# -------------------------------
# Visualizing Coffee Badgers Over Time
# -------------------------------

# Aggregate the number of total employees and coffee badgers per day
daily_badges = df_badge.groupby("Day").size()
daily_coffee_badgers = df_badge[df_badge["Coffee_Badger"] == 1].groupby("Day").size()

# Plot coffee badgers and normal employees
plt.figure(figsize=(12, 6))
plt.plot(daily_badges.index, daily_badges.values, color="black", linewidth=1, markersize=2, 
         markerfacecolor='white', markeredgecolor='black', label="Total Employees Badged In")
plt.plot(daily_coffee_badgers.index, daily_coffee_badgers.values, color="red", linestyle="dashed", linewidth=1, 
          markersize=2, markerfacecolor='white', markeredgecolor='red', label="Coffee Badgers")

plt.xlabel("Day", fontsize=12)
plt.ylabel("Number of Employees", fontsize=12)
plt.title("Daily Badge-Ins with Coffee Badgers", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Extend left spine downward slightly for bracket effect
plt.gca().spines["left"].set_bounds(min(daily_badges.values), max(daily_badges.values))
plt.gca().spines["bottom"].set_bounds(min(daily_badges.index), max(daily_badges.index))

plt.legend(frameon=False)
plt.savefig("coffee_badger_time_series.png")
plt.show()

# -------------------------------
# Building a Time Series Classification Model
# -------------------------------

# Create feature set for classification
df_badge["Duration"] = df_badge["Badge_Out"] - df_badge["Badge_In"]
features = df_badge[["Badge_In", "Duration"]]
labels = df_badge["Coffee_Badger"]

# Split into train and test sets using time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(features))[ -1 ]
X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classification_results = classification_report(y_test, y_pred)

logger.info("Classification Report for Coffee Badgers Detection:")
logger.info(classification_results)
