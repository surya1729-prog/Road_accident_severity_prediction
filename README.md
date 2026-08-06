# Traffic-Accident-Severity-and-Hotspot-Detection

Live Demo: [Click Here](https://road-accident-analysis-severity-prediction-xqlzia5xtupo7fjzczm.streamlit.app/)

A data-driven exploration of 100,000+ US road accidents that uncovers hidden patterns in time, weather, and geography — and then applies Machine Learning to predict accident severity and map high-risk hotspots across the country.

End-to-end ML pipeline on US Accidents dataset to detect road accident hotspots and predict severity. Uses Random Forest, XGBoost &amp; Logistic Regression with SMOTE balancing, K-Means clustering, Folium geospatial maps, Plotly trends and a Streamlit dashboard. Built on Google Colab.

# US Road Accident Analysis & Severity Prediction

## Introduction

Most road safety projects jump straight to modelling. This one doesn't. The analysis comes first - understanding *when*, *where*, and *under what conditions* accidents happen - and the ML models are built on top of those findings to make the insights actionable.

## Data Analysis

Before any model is trained, the data is thoroughly explored:

- **Temporal Patterns** - When do accidents peak? Rush hours (7–9 AM, 5–7 PM) account for a disproportionate share of incidents. Weekdays see significantly more accidents than weekends.
- **Weather Impact** - Low visibility and extreme temperatures correlate strongly with higher severity accidents.
- **Geographic Distribution** - Accidents are not uniformly spread; certain states and corridors are heavily overrepresented.
- **Severity Breakdown** - The majority of accidents fall in the high-severity range, making class balancing a critical step.

---

## ML

With the patterns established, ML is used to formalize and extend the findings:

| Task | Approach |
|------|----------|
| Severity Prediction | Random Forest, XGBoost, Logistic Regression |
| Class Imbalance | SMOTE Balancing |
| Hotspot Detection | K-Means Clustering (Elbow Method) |
| Spatial Mapping | Folium (Heatmap, Cluster Map, Severity Map) |

Best model accuracy: **~92% (Random Forest)**

---

## Project Workflow

| Step | Description |
|------|-------------|
| 1 | Data Loading & Initial Inspection |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Preprocessing & Feature Engineering |
| 4 | Trend Analysis (Plotly) |
| 5 | Feature Selection & SMOTE Balancing |
| 6 | Model Training & Evaluation |
| 7 | K-Means Hotspot Clustering |
| 8 | Geospatial Visualisation (Folium) |
| 9 | Saving Outputs |
| 10 | Streamlit Dashboard |

---

## Key Features Engineered

From raw timestamps and sensor data, meaningful features were extracted:

- **Temporal:** Hour, DayOfWeek, Month, IsWeekend, IsRushHour, IsNight
- **Weather:** Temperature, Visibility, Wind Speed, Humidity, Precipitation
- **Infrastructure:** Junction, Traffic Signal, Crossing, Roundabout, Railway
- **Geospatial:** Start_Lat, Start_Lng

---

## Geospatial Outputs

Three interactive Folium maps generated:

- **Accident Heatmap** - density visualization across the US
- **K-Means Cluster Map** - 6 identified high-risk regions with accident counts
- **Severity Map** - color-coded markers (Green = Minor → Red = Fatal)

---

## Streamlit Dashboard

A 5-page interactive dashboard that brings the entire project together:

1. **Overview** - Project summary and tech stack
2. **EDA & Trends** - Severity distribution, hourly and daily patterns
3. **Model Results** - Accuracy comparison and confusion matrices
4. **Severity Predictor** - Live prediction using input sliders
5. **Hotspot Clusters** - Cluster table and Folium map links

Deployed live from Google Colab via ngrok.

---

## Dataset

**US Accidents (March 2023)** - [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

- 7.7 million records across the US (2016–2023)
- 100,000 rows sampled for this project
- 46 features covering location, weather, road type, and time

---

## Technology Used

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Analysis | Pandas, Scikit-learn, XGBoost |
| Balancing | SMOTE (imbalanced-learn) |
| Clustering | K-Means |
| Visualisation | Plotly, Folium, Matplotlib |
| ML | K-Mean, Random-Forest, Logistic Regression, XGBoost |
| Dashboard | Streamlit |
| Platform | Google Colab |
