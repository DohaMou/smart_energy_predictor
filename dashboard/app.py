
# ============================================================
# SMART ENERGY PREDICTOR - CALAIS
# Professional Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Energy Predictor | Calais",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 4px;
}

h1 {
    color: #102A43;
    font-weight: 800;
}

h2, h3 {
    color: #243B53;
}

p {
    color: #486581;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(16,42,67,0.05);
}

[data-testid="stMetricLabel"] {
    color: #627D98;
}

[data-testid="stMetricValue"] {
    color: #1565C0;
    font-weight: 700;
}

[data-testid="stPlotlyChart"] {
    background: white;
    padding: 10px;
    border-radius: 14px;
}

div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}

hr {
    border-color: #D9E2EC;
}

.footer {
    text-align: center;
    color: #829AB1;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. CONSTANTS
# ============================================================

DATA_PATH = Path("data/processed/features_2022.csv")

PREDICTIONS_PATH = Path(
    "data/processed/predictions_test.csv"
)

FEATURE_COLUMNS = [
    "temperature_c",
    "hour",
    "day_of_week",
    "day",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h"
]


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["ts"] = pd.to_datetime(
        df["ts"], utc=True
    )

    df["local_date"] = (
        df["ts"].dt.tz_convert("Europe/Paris")
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"], utc=True
    )

    return df, predictions


try:
    df, predictions = load_data()

except FileNotFoundError as error:

    st.error(f"Required data file not found: {error}")
    st.info(
        "Run the feature engineering and model evaluation scripts first."
    )
    st.stop()


# ============================================================
# 5. GLOBAL STYLE
# ============================================================

CHART_LAYOUT = {
    "template": "plotly_white",
    "font": {
        "family": "Arial",
        "color": "#243B53"
    },
    "title": {
        "x": 0.03,
        "font": {"size": 19}
    },
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": dict(l=30, r=30, t=65, b=35)
}


def style_chart(fig):

    fig.update_layout(**CHART_LAYOUT)

    return fig


def metric_card(label, value, help_text=None):

    st.metric(
        label=label,
        value=value,
        help=help_text
    )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("# ⚡ Smart Energy")

    st.caption("ENERGY INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Consumption",
            "Weather Analysis",
            "Prediction",
            "Model Performance",
            "Anomaly Detection",
            "About the Project"
        ]
    )

    st.divider()

    st.markdown("### DATA INFORMATION")

    st.caption("Year: 2022")
    st.caption("Region: Hauts-de-France")
    st.caption("Weather proxy: Calais")
    st.caption("Model: XGBoost")

    st.divider()

    st.caption("Smart Energy Predictor")
    st.caption("Portfolio Project | 2026")


# ============================================================
# 7. GLOBAL METRICS
# ============================================================

average = df["consumption_per_meter"].mean()

maximum = df["consumption_per_meter"].max()

minimum = df["consumption_per_meter"].min()

observations = len(df)

correlation = df[
    ["temperature_c", "consumption_per_meter"]
].corr().iloc[0, 1]


# ============================================================
# 8. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("⚡ Smart Energy Predictor")

    st.subheader("Energy Consumption Dashboard")

    st.write(
        "Monitor, analyze and understand residential "
        "energy consumption using Machine Learning."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Average Consumption",
            f"{average:.3f} kWh"
        )

    with col2:
        metric_card(
            "Maximum Consumption",
            f"{maximum:.3f} kWh"
        )

    with col3:
        metric_card(
            "Minimum Consumption",
            f"{minimum:.3f} kWh"
        )

    with col4:
        metric_card(
            "Observations",
            f"{observations:,}"
        )

    st.divider()

    st.subheader("Daily Energy Consumption")

    daily = (
        df.set_index("local_date")[
            "consumption_per_meter"
        ]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig = px.area(
        daily,
        x="local_date",
        y="consumption_per_meter",
        labels={
            "local_date": "Date",
            "consumption_per_meter":
                "Consumption (kWh/meter/hour)"
        }
    )

    fig.update_traces(
        line_color="#1976D2",
        fillcolor="rgba(25,118,210,0.12)"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Consumption by Month")

        monthly = (
            df.groupby("month")[
                "consumption_per_meter"
            ]
            .mean()
            .reset_index()
        )

        fig_month = px.bar(
            monthly,
            x="month",
            y="consumption_per_meter",
            labels={
                "month": "Month",
                "consumption_per_meter": "Average Consumption"
            }
        )

        fig_month.update_traces(
            marker_color="#1976D2"
        )

        st.plotly_chart(
            style_chart(fig_month),
            use_container_width=True
        )

    with col2:

        st.subheader("Consumption Distribution")

        fig_hist = px.histogram(
            df,
            x="consumption_per_meter",
            nbins=40,
            labels={
                "consumption_per_meter":
                    "Consumption (kWh/meter/hour)"
            }
        )

        fig_hist.update_traces(
            marker_color="#00A6A6"
        )

        st.plotly_chart(
            style_chart(fig_hist),
            use_container_width=True
        )

    st.info(
        "Geographic scope: Enedis residential consumption "
        "averages for Hauts-de-France. Calais weather is used "
        "as a proxy and does not represent Calais-only consumption."
    )


# ============================================================
# 9. CONSUMPTION PAGE
# ============================================================

elif page == "Consumption":

    st.title("Energy Consumption Analysis")

    st.write(
        "Explore consumption patterns over time."
    )

    min_date = df["local_date"].min().date()

    max_date = df["local_date"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered = df[
            (df["local_date"].dt.date >= start_date)
            & (df["local_date"].dt.date <= end_date)
        ]

    else:
        filtered = df

    daily = (
        filtered.set_index("local_date")[
            "consumption_per_meter"
        ]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig = px.line(
        daily,
        x="local_date",
        y="consumption_per_meter",
        title="Daily Consumption Trend"
    )

    fig.update_traces(
        line_color="#1976D2"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        hourly = (
            filtered.groupby("hour")[
                "consumption_per_meter"
            ]
            .mean()
            .reset_index()
        )

        fig_hour = px.line(
            hourly,
            x="hour",
            y="consumption_per_meter",
            title="Average Consumption by Hour"
        )

        fig_hour.update_traces(
            line_color="#00A6A6"
        )

        st.plotly_chart(
            style_chart(fig_hour),
            use_container_width=True
        )

    with col2:

        weekday = (
            filtered.groupby("day_of_week")[
                "consumption_per_meter"
            ]
            .mean()
            .reset_index()
        )

        fig_week = px.bar(
            weekday,
            x="day_of_week",
            y="consumption_per_meter",
            title="Consumption by Day of Week"
        )

        fig_week.update_traces(
            marker_color="#1976D2"
        )

        st.plotly_chart(
            style_chart(fig_week),
            use_container_width=True
        )

    st.metric(
        "Average in selected period",
        f"{filtered['consumption_per_meter'].mean():.3f} kWh"
        if not filtered.empty else "No data"
    )


# ============================================================
# 10. WEATHER ANALYSIS PAGE
# ============================================================

elif page == "Weather Analysis":

    st.title("Weather & Energy Analysis")

    st.write(
        "Understand the relationship between temperature "
        "and energy consumption."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Temperature Correlation",
            f"{correlation:.3f}"
        )

    with col2:

        st.metric(
            "Average Temperature",
            f"{df['temperature_c'].mean():.1f} °C"
        )

    fig = px.scatter(
        df,
        x="temperature_c",
        y="consumption_per_meter",
        opacity=0.35,
        trendline="ols",
        title="Temperature vs Energy Consumption",
        labels={
            "temperature_c": "Temperature (°C)",
            "consumption_per_meter":
                "Consumption (kWh/meter/hour)"
        }
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    monthly_temp = (
        df.groupby("month")[
            ["temperature_c", "consumption_per_meter"]
        ]
        .mean()
        .reset_index()
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=monthly_temp["month"],
            y=monthly_temp["temperature_c"],
            name="Temperature",
            line=dict(color="#E67E22")
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=monthly_temp["month"],
            y=monthly_temp["consumption_per_meter"],
            name="Consumption",
            yaxis="y2",
            line=dict(color="#1976D2")
        )
    )

    fig2.update_layout(
        title="Monthly Temperature and Consumption",
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(
            title="Consumption",
            overlaying="y",
            side="right"
        )
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

    st.info(
        "Correlation measures association, not causation. "
        "Weather observations are from Calais and consumption "
        "is a regional Hauts-de-France average."
    )


# ============================================================
# 11. PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.title("Energy Consumption Prediction")

    st.write(
        "Compare the real consumption with the XGBoost predictions."
    )

    fig = px.line(
        predictions,
        x="Date",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted Consumption",
        labels={
            "value": "kWh/meter/hour",
            "Date": "Date",
            "variable": "Series"
        }
    )

    fig.update_traces(line_width=2)

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    predictions_display = predictions.copy()

    predictions_display["Error"] = (
        predictions_display["Actual"]
        - predictions_display["Predicted"]
    )

    predictions_display["Absolute Error"] = (
        predictions_display["Error"].abs()
    )

    st.subheader("Prediction Details")

    st.dataframe(
        predictions_display.tail(30),
        use_container_width=True
    )

    st.download_button(
        "Download predictions CSV",
        data=predictions_display.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="energy_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# 12. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.title("Machine Learning Model Performance")

    actual = predictions["Actual"]

    predicted = predictions["Predicted"]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual,
        predicted
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("MAE", f"{mae:.4f}")

    with col2:
        metric_card("RMSE", f"{rmse:.4f}")

    with col3:
        metric_card("R² Score", f"{r2:.4f}")

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Training split:** 70%")
    st.write("**Validation split:** 15%")
    st.write("**Testing split:** 15%")
    st.write("**Test observations:**", len(predictions))

    st.subheader("Prediction Error Distribution")

    errors = actual - predicted

    fig = px.histogram(
        x=errors,
        nbins=40,
        title="Distribution of Prediction Errors",
        labels={"x": "Actual - Predicted"}
    )

    fig.update_traces(
        marker_color="#1976D2"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.success(
        "The XGBoost model has been evaluated on "
        "the chronological test set."
    )


# ============================================================
# 13. ANOMALY DETECTION PAGE
# ============================================================

elif page == "Anomaly Detection":

    st.title("Energy Consumption Anomaly Detection")

    st.write(
        "Identify unusually high or low consumption values "
        "using an exploratory statistical method."
    )

    # IQR method: exploratory flags, not confirmed faults
    q1 = df["consumption_per_meter"].quantile(0.25)

    q3 = df["consumption_per_meter"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    anomalies = df[
        (df["consumption_per_meter"] < lower_bound)
        | (df["consumption_per_meter"] > upper_bound)
    ].copy()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Flagged observations",
            f"{len(anomalies):,}"
        )

    with col2:
        metric_card(
            "Lower threshold",
            f"{lower_bound:.3f}"
        )

    with col3:
        metric_card(
            "Upper threshold",
            f"{upper_bound:.3f}"
        )

    fig = px.scatter(
        df,
        x="local_date",
        y="consumption_per_meter",
        title="Consumption and Potential Anomalies"
    )

    if not anomalies.empty:

        fig.add_trace(
            go.Scatter(
                x=anomalies["local_date"],
                y=anomalies["consumption_per_meter"],
                mode="markers",
                name="Potential anomalies",
                marker=dict(
                    color="red",
                    size=7,
                    symbol="x"
                )
            )
        )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.subheader("Flagged observations")

    st.dataframe(
        anomalies[
            [
                "local_date",
                "temperature_c",
                "consumption_per_meter"
            ]
        ],
        use_container_width=True
    )

    st.warning(
        "These are statistical outliers flagged by the IQR method. "
        "They are not necessarily equipment faults or abnormal "
        "consumer behavior."
    )


# ============================================================
# 14. ABOUT THE PROJECT PAGE
# ============================================================

elif page == "About the Project":

    st.title("About Smart Energy Predictor")

    st.write(
        "Smart Energy Predictor is a Data Science and "
        "Machine Learning project designed to analyze "
        "residential energy consumption patterns and "
        "evaluate hourly consumption predictions."
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze residential electricity consumption patterns.
    - Explore the relationship between weather and energy use.
    - Build predictive models using Machine Learning.
    - Evaluate prediction accuracy on unseen chronological data.
    - Identify potential consumption anomalies.
    """)

    st.divider()

    st.subheader("Data Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Electricity Consumption")
        st.write("**Source:** Enedis")
        st.write("**Geographic area:** Hauts-de-France")
        st.write("**Period:** 2022")
        st.write("**Granularity:** Hourly")
        st.write("**Population:** Residential consumption")

    with col2:
        st.markdown("### Weather Data")
        st.write("**Location:** Calais, France")
        st.write("**Period:** 2022")
        st.write("**Variable:** Air temperature")
        st.write("**Granularity:** Hourly")

    st.info(
        "Important geographic limitation: electricity "
        "consumption represents a regional residential "
        "average for Hauts-de-France, not Calais alone. "
        "Calais weather observations are used as a "
        "weather proxy."
    )

    st.divider()

    st.subheader("Machine Learning Methodology")

    st.markdown("""
    **Algorithm:** XGBoost Regressor

    **Feature Engineering:**
    - Calendar and time features
    - Temperature
    - Weekend and public holiday indicators
    - Hourly, daily and weekly lag features
    - 24-hour rolling average

    **Chronological Data Split:**
    - Training: 70%
    - Validation: 15%
    - Testing: 15%
    """)

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "0.0200")
    col2.metric("RMSE", "0.0276")
    col3.metric("R² Score", "0.9679")

    st.caption(
        "Metrics are calculated on the chronological "
        "2022 test set containing 1,289 observations."
    )

    st.divider()

    st.subheader("Technologies Used")

    st.markdown("""
    - **Python** — Data processing and modeling
    - **Pandas & NumPy** — Data analysis
    - **Scikit-learn** — Model evaluation
    - **XGBoost** — Machine Learning
    - **Plotly** — Interactive visualizations
    - **Streamlit** — Dashboard development
    - **PostgreSQL** — Data storage during the pipeline
    - **Git & GitHub** — Version control
    """)

    st.divider()

    st.subheader("Limitations")

    st.markdown("""
    - The analysis covers the year 2022 only.
    - Consumption data is regional, not city-specific.
    - Weather data from Calais is used as a proxy.
    - Predictions are evaluated on a historical test set,
      not as live future forecasts.
    - Anomaly detection uses the exploratory IQR statistical
      method. Flagged observations are not confirmed faults.
    """)

    st.divider()

    st.subheader("Project Summary")

    st.success(
        "Smart Energy Predictor demonstrates an end-to-end "
        "Data Science workflow, from data ingestion and "
        "feature engineering to Machine Learning evaluation "
        "and interactive dashboard deployment."
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Smart Energy Predictor - Calais<br>
    Data Science & Machine Learning Portfolio Project<br>
    Enedis regional consumption data | 2022
</div>
""", unsafe_allow_html=True)
