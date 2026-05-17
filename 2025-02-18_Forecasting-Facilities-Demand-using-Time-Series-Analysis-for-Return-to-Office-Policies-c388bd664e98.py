import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA


def plot_daily_badge_in_counts(daily_badges) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(daily_badges.index, daily_badges.values, color="black", label="Employees Badged In")

    plt.xlabel("Day")

    plt.ylabel("Number of Employees in Office")

    plt.title("Daily Employee Badge-Ins Over 300 Days")

    plt.legend()

    plt.savefig("badge_data_time_series.png")

    plt.show()

    arima_model = ARIMA(daily_badges, order=(2, 1, 2)).fit()

    forecast = arima_model.forecast(30)


def plot_actual_data_and_forecast(daily_badges, forecast, n_days) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(daily_badges.index, daily_badges.values, label="Actual Attendance", color="black")

    plt.plot(
        range(n_days, n_days + 30),
        forecast,
        label="Forecasted Attendance",
        color="red",
        linestyle="dashed",
    )

    plt.xlabel("Day")

    plt.ylabel("Number of Employees in Office")

    plt.title("ARIMA Forecast for Future Badge-Ins")

    plt.legend()

    plt.savefig("badge_in_forecast.png")

    plt.show()


def plot_coffee_badgers_and_normal_employees(daily_badges, daily_coffee_badgers) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_badges.index,
        daily_badges.values,
        color="black",
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
        markeredgecolor="black",
        label="Total Employees Badged In",
    )

    plt.plot(
        daily_coffee_badgers.index,
        daily_coffee_badgers.values,
        color="red",
        linestyle="dashed",
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
        markeredgecolor="red",
        label="Coffee Badgers",
    )

    plt.xlabel("Day", fontsize=12)

    plt.ylabel("Number of Employees", fontsize=12)

    plt.title("Daily Badge-Ins with Coffee Badgers", fontsize=14)

    plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)

    plt.gca().spines["left"].set_bounds(min(daily_badges.values), max(daily_badges.values))

    plt.gca().spines["bottom"].set_bounds(min(daily_badges.index), max(daily_badges.index))

    plt.legend(frameon=False)

    plt.savefig("coffee_badger_time_series.png")

    plt.show()


def create_feature_set_for_classification(df_badge, logger) -> None:
    df_badge["Duration"] = df_badge["Badge_Out"] - df_badge["Badge_In"]

    features = df_badge[["Badge_In", "Duration"]]

    labels = df_badge["Coffee_Badger"]

    tscv = TimeSeriesSplit(n_splits=5)

    train_idx, test_idx = list(tscv.split(features))[-1]

    X_train, X_test = (features.iloc[train_idx], features.iloc[test_idx])

    y_train, y_test = (labels.iloc[train_idx], labels.iloc[test_idx])

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    classification_results = classification_report(y_test, y_pred)

    logger.info("Classification Report for Coffee Badgers Detection:")

    logger.info(classification_results)


def main() -> None:
    np.random.seed(42)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    n_employees = 1000

    n_days = 300

    badge_data = []

    for emp_id in range(1, n_employees + 1):
        for day in range(1, n_days + 1):
            if np.random.rand() < 0.6:
                badge_in_time = np.random.normal(loc=9, scale=1)
                badge_out_time = badge_in_time + np.random.normal(loc=8, scale=1.5)
                badge_data.append(
                    [emp_id, day, max(7, min(11, badge_in_time)), max(15, min(20, badge_out_time))]
                )

    df_badge = pd.DataFrame(badge_data, columns=["Employee_ID", "Day", "Badge_In", "Badge_Out"])

    daily_badges = df_badge.groupby("Day").size()

    signalplot.apply(font_family="serif")

    n_employees = 1000

    n_days = 300

    badge_data = []

    for emp_id in range(1, n_employees + 1):
        for day in range(1, n_days + 1):
            if np.random.rand() < 0.6:
                badge_in_time = np.random.normal(loc=9, scale=1)
                badge_out_time = badge_in_time + np.random.normal(loc=8, scale=1.5)
                badge_data.append(
                    [emp_id, day, max(7, min(11, badge_in_time)), max(15, min(20, badge_out_time))]
                )

    df_badge = pd.DataFrame(badge_data, columns=["Employee_ID", "Day", "Badge_In", "Badge_Out"])

    coffee_badger_percentage = 0.1

    coffee_badger_ids = np.random.choice(
        df_badge["Employee_ID"].unique(),
        size=int(n_employees * coffee_badger_percentage),
        replace=False,
    )

    df_badge.loc[df_badge["Employee_ID"].isin(coffee_badger_ids), "Badge_Out"] = df_badge.loc[
        df_badge["Employee_ID"].isin(coffee_badger_ids), "Badge_In"
    ] + np.random.uniform(
        0.2, 1.0, size=len(df_badge[df_badge["Employee_ID"].isin(coffee_badger_ids)])
    )

    df_badge["Coffee_Badger"] = df_badge.apply(
        lambda row: 1 if row["Badge_Out"] - row["Badge_In"] < 1 else 0, axis=1
    )

    daily_badges = df_badge.groupby("Day").size()

    daily_coffee_badgers = df_badge[df_badge["Coffee_Badger"] == 1].groupby("Day").size()
    plot_daily_badge_in_counts(daily_badges)
    plot_actual_data_and_forecast(daily_badges, forecast, n_days)
    plot_coffee_badgers_and_normal_employees(daily_badges, daily_coffee_badgers)
    create_feature_set_for_classification(df_badge, logger)


if __name__ == "__main__":
    main()
