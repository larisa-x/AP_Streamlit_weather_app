import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
from datetime import datetime

WINDOW = 30

month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

SEASONS_ORDER = ["winter", "spring", "summer", "autumn"]


def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("temperature_data.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["temperature"] = pd.to_numeric(df["temperature"])
    if "season" not in df.columns:
        df["season"] = df["timestamp"].dt.month.map(lambda x: month_to_season[x])
    return df


def add_rolling_and_anomalies(df_city):
    df_city = df_city.sort_values("timestamp").copy()

    df_city["rolling_mean"] = (
        df_city["temperature"]
        .rolling(window=WINDOW, min_periods=WINDOW // 2)
        .mean()
    )

    sigma = df_city["temperature"].std(ddof=0)

    df_city["upper"] = df_city["rolling_mean"] + 2 * sigma
    df_city["lower"] = df_city["rolling_mean"] - 2 * sigma

    df_city["is_anomaly"] = (
        (df_city["temperature"] > df_city["upper"]) |
        (df_city["temperature"] < df_city["lower"])
    )

    return df_city, float(sigma)


def seasonal_statistics(df_city):
    stats = (
        df_city
        .groupby("season")["temperature"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["season"] = pd.Categorical(stats["season"], categories=SEASONS_ORDER, ordered=True)
    stats = stats.sort_values("season")
    return stats


def plot_time_series(df_city):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df_city["timestamp"],
        df_city["temperature"],
        label="Температура",
        linewidth=0.7,
        alpha=0.5,
    )

    ax.plot(
        df_city["timestamp"],
        df_city["rolling_mean"],
        label="Скользящее среднее (30 дней)",
        linewidth=2,
    )

    ax.plot(df_city["timestamp"], df_city["upper"], label="+2σ")
    ax.plot(df_city["timestamp"], df_city["lower"], label="-2σ")

    anomalies = df_city[df_city["is_anomaly"]]
    ax.scatter(
        anomalies["timestamp"],
        anomalies["temperature"],
        s=15,
        label="Аномалии",
    )

    ax.set_title("Временной ряд и аномалии")
    ax.set_xlabel("Дата")
    ax.set_ylabel("°C")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_trend(df_city):
    yearly = (
        df_city
        .set_index("timestamp")["temperature"]
        .resample("YS")
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values, label="Средняя по году")

    if len(yearly) > 1:
        x = np.arange(len(yearly))
        slope, intercept = np.polyfit(x, yearly.values, 1)
        trend = intercept + slope * x
        ax.plot(yearly.index, trend, label="Линейный тренд")

    ax.set_title("Долгосрочный тренд")
    ax.set_xlabel("Год")
    ax.set_ylabel("°C")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def get_current_temperature(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if r.status_code == 401:
        return None, data

    if r.status_code != 200:
        return None, {"cod": r.status_code, "message": data.get("message", "API error")}

    return float(data["main"]["temp"]), None


def seasonal_norm(df_city, season):
    part = df_city[df_city["season"] == season]["temperature"]
    mean = float(part.mean())
    std = float(part.std(ddof=0))
    return mean - 2 * std, mean + 2 * std, mean, std


def cities_volatility_ranking(df):
    stats = (
        df
        .groupby("city")["temperature"]
        .agg(mean="mean", std=lambda x: x.std(ddof=0), min="min", max="max", n="count")
        .reset_index()
    )
    stats["std"] = stats["std"].astype(float)
    stats = stats.sort_values("std", ascending=False).reset_index(drop=True)
    return stats


def highlight_selected_city(row, selected_city):
    if row["city"] == selected_city:
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)


def main():
    st.set_page_config(layout="wide")
    st.title("Анализ температурных данных и мониторинг текущей температуры")

    st.sidebar.header("Настройки")
    uploaded_file = st.sidebar.file_uploader("Загрузить CSV с историческими данными", type="csv")

    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error("Не удалось прочитать данные.")
        st.code(str(e))
        return

    cities = sorted(df["city"].unique())
    if len(cities) == 0:
        st.error("В данных нет городов.")
        return

    city = st.sidebar.selectbox("Город", cities)

    df_city = df[df["city"] == city].copy()
    df_city, sigma_city = add_rolling_and_anomalies(df_city)

    c1, c2, c3 = st.columns(3)
    c1.metric("Наблюдений", len(df_city))
    c2.metric("Аномалий", int(df_city["is_anomaly"].sum()))
    c3.metric("σ (город)", f"{sigma_city:.2f} °C")

    st.subheader("Сезонная статистика")
    season_stats = seasonal_statistics(df_city).copy()
    season_stats["mean"] = season_stats["mean"].round(2)
    season_stats["std"] = season_stats["std"].round(2)
    st.dataframe(season_stats, use_container_width=True)

    st.subheader("Временной ряд и аномалии")
    st.pyplot(plot_time_series(df_city), clear_figure=True)

    st.subheader("Таблица аномалий (последние 50)")
    anomalies = df_city[df_city["is_anomaly"]][
        ["timestamp", "temperature", "rolling_mean", "upper", "lower"]
    ]
    st.dataframe(anomalies.tail(50), use_container_width=True)

    st.subheader("Долгосрочный тренд")
    st.pyplot(plot_trend(df_city), clear_figure=True)

    st.subheader("Бонус: рейтинг городов по изменчивости температуры")
    ranking = cities_volatility_ranking(df).copy()
    ranking["mean"] = ranking["mean"].round(2)
    ranking["std"] = ranking["std"].round(2)
    ranking["min"] = ranking["min"].round(2)
    ranking["max"] = ranking["max"].round(2)

    styled = ranking.style.apply(
        highlight_selected_city,
        axis=1,
        selected_city=city
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()
    st.header("Мониторинг текущей температуры через OpenWeatherMap")

    api_key = st.text_input("OpenWeatherMap API key", type="password")

    if api_key and st.button("Получить текущую температуру"):
        temp, error = get_current_temperature(city, api_key)

        if error is not None:
            st.error("Ошибка OpenWeatherMap")
            st.json(error)
        else:
            st.success(f"Текущая температура в {city}: {temp:.1f} °C")

            season_now = month_to_season[datetime.utcnow().month]
            low, high, mean, std = seasonal_norm(df_city, season_now)

            st.write(f"Сезон сейчас: **{season_now}**")
            st.write(f"Нормальный диапазон по истории: **[{low:.1f}, {high:.1f}] °C**")

            if temp < low or temp > high:
                st.error("Температура выходит за нормальный диапазон сезона.")
            else:
                st.info("Температура в пределах нормального диапазона сезона.")


if __name__ == "__main__":
    main()
