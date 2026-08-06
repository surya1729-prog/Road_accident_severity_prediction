import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(
    page_title='Road Accident Hotspot Detection',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Dark theme CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; color: white; }
[data-testid="stSidebar"]          { background-color: #1a1d27; }
.metric-box   { background:#1e2130; border-radius:8px; padding:18px 20px; margin:5px; }
.metric-label { color:#aaa; font-size:13px; }
.metric-value { color:white; font-size:26px; font-weight:bold; }
.footer { color:#555; font-size:12px; text-align:center;
           border-top:1px solid #333; padding-top:12px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

FOOTER = "<div class='footer'>Road Accident Hotspot Detection | Surya Charan </div>"
SAVE_DIR = 'Outputs/'

# Load saved artefacts
@st.cache_resource
def load_artefacts():
    with open(SAVE_DIR + 'best_model.pkl', 'rb') as f: best_model     = pickle.load(f)
    with open(SAVE_DIR + 'scaler.pkl',     'rb') as f: scaler         = pickle.load(f)
    with open(SAVE_DIR + 'kmeans.pkl',     'rb') as f: kmeans         = pickle.load(f)
    cluster_summary = pd.read_csv(SAVE_DIR + 'cluster_summary.csv')
    return best_model, scaler, kmeans, cluster_summary

try:
    best_model, scaler, kmeans, cluster_summary = load_artefacts()
    artefacts_ok = True
except Exception as e:
    artefacts_ok  = False
    artefact_err  = str(e)

# Dark-styled matplotlib axes helper
def dark_ax(ax):
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ax.spines.values(): sp.set_color('#444')

# Sidebar navigation
with st.sidebar:
    st.markdown('### Road Accident Hotspot')
    st.markdown('Surya Charan')
    st.markdown('---')
    page = st.radio('Navigate', [
        'Overview',
        'EDA & Trends',
        'Model Results & Confusion Matrix',
        'Severity Predictor',
        'Hotspot Clusters',
    ])

# ================================================================
# PAGE 1 — OVERVIEW
# ================================================================
if page == 'Overview':
    st.title('Road Accident Hotspot Detection')
    st.markdown('---')

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ['Dataset', 'Features', 'ML Models', 'Hotspot Clusters'],
        ['US Accidents 2023', '32', '3 (RF · XGB · LR)', 'K=6 (KMeans)']
    ):
        col.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(' Modules Completed')
        for m in [
            ' Data Loading & Preprocessing',
            ' Exploratory Data Analysis (Plotly)',
            ' SMOTE Class Balancing',
            ' Severity Prediction — RF · XGB · LR',
            ' Confusion Matrices (all models)',
            ' KMeans Hotspot Clustering (K=6)',
            ' Geospatial Maps (Folium)',
            ' All outputs saved to Drive',
        ]:
            st.markdown(f'- {m}')
    with c2:
        st.subheader(' Tech Stack')
        tech = pd.DataFrame({
            'Component':  ['Language','ML','Clustering','Viz','Dashboard','Platform'],
            'Technology': ['Python 3','RF · XGBoost · LR','KMeans (K=6)',
                           'Plotly · Folium','Streamlit','Google Colab']
        })
        st.dataframe(tech, use_container_width=True, hide_index=True)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ================================================================
# PAGE 2 — EDA & TRENDS
# ================================================================
elif page == 'EDA & Trends':
    st.title(' Exploratory Data Analysis')
    st.markdown('---')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Severity Distribution')
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1e2130')
        sizes  = [45.4, 30.5, 24.1]
        labels = ['High Severity', 'Medium', 'Low']
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90,
               textprops={'color':'white', 'fontsize':12})
        ax.set_facecolor('#1e2130')
        st.pyplot(fig)

    with c2:
        st.subheader('Accidents by Hour of Day')
        hours = list(range(24))
        vals  = [120,80,60,50,70,150,280,340,300,260,
                 240,250,270,260,250,280,340,370,320,
                 280,240,200,170,140]
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#1e2130')
        dark_ax(ax)
        ax.plot(hours, vals, color='#3498db', linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, vals, alpha=0.3, color='#3498db')
        ax.axvspan(7,  9,  alpha=0.15, color='red', label='Rush Hours')
        ax.axvspan(17, 19, alpha=0.15, color='red')
        ax.set_xlabel('Hour'); ax.set_ylabel('Accidents')
        ax.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('Accidents by Day of Week')
    days     = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    day_vals = [15200,15800,15500,15300,16100,10200,9700]
    fig, ax  = plt.subplots(figsize=(9,4), facecolor='#1e2130')
    dark_ax(ax)
    ax.bar(days, day_vals, color='#9b59b6')
    ax.set_ylabel('Number of Accidents')
    for i,v in enumerate(day_vals):
        ax.text(i, v+100, f'{v:,}', ha='center', fontsize=8, color='white')
    st.pyplot(fig)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ================================================================
# PAGE 3 — MODEL RESULTS + CONFUSION MATRICES
# ================================================================
elif page == 'Model Results & Confusion Matrix':
    st.title(' Model Results & Confusion Matrices')
    st.markdown('---')

    # Accuracy cards
    c1, c2, c3 = st.columns(3)
    for col, (name, acc, color) in zip(
        [c1, c2, c3],
        [('Random Forest','~82.05%','#2ecc71'),
         ('XGBoost',      '~90.20%','#3498db'),
         ('Logistic Reg', '~60.87%','#e67e22')]
    ):
        col.markdown(f"""
        <div class='metric-box' style='border-left:4px solid {color}'>
            <div class='metric-label'>{name}</div>
            <div class='metric-value' style='color:{color}'>{acc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('---')

    # Accuracy bar chart
    st.subheader('Accuracy Comparison')
    names = ['Random Forest','XGBoost','Logistic Regression']
    accs  = [0.8205, 0.9020, 0.6087]
    fig, ax = plt.subplots(figsize=(8,4), facecolor='#1e2130')
    dark_ax(ax)
    bar_colors = ['#2ecc71' if a == max(accs) else '#3498db' for a in accs]
    bars = ax.bar(names, [a*100 for a in accs], color=bar_colors, edgecolor='black')
    ax.set_ylim(0,110); ax.set_ylabel('Accuracy (%)')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+1,
                f'{acc*100:.1f}%', ha='center', color='white', fontweight='bold')
    st.pyplot(fig)

    st.markdown('---')

    # Confusion matrices — use saved PNG if available
    st.subheader('Confusion Matrices (All Models)')
    cm_path = SAVE_DIR + 'confusion_matrices.png'
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True)
    else:
        st.info('ℹ️ Run the **Step 11 (Confusion Matrix)** cell in the notebook first, then relaunch this app.')
        # Fallback — synthetic demo so page is not blank
        fig, axes = plt.subplots(1, 3, figsize=(15,5), facecolor='#1e2130')
        fig.suptitle('Confusion Matrices (demo)', color='white', fontsize=13)
        cms = [
            np.array([[8200, 620],[580, 9400]]),
            np.array([[8100, 720],[650, 9330]]),
            np.array([[7000,1800],[1600,8400]]),
        ]
        for ax, cm, nm in zip(axes, cms, names):
            disp = ConfusionMatrixDisplay(cm, display_labels=['Low','High'])
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(nm, color='white')
            ax.set_facecolor('#1e2130')
            for item in ([ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ================================================================
# PAGE 4 — SEVERITY PREDICTOR
# ================================================================

elif page == 'Severity Predictor':
    st.title(' Accident Severity Predictor')
    st.markdown('Fill in conditions to predict accident severity using the best trained model.')
    st.markdown('---')

    if not artefacts_ok:
        st.warning(f'Could not load saved models: {artefact_err}\n\nRun the full notebook first.')
    else:
        # ── Numerical Features (10) ───────────────────────────────
        st.markdown('#### Weather & Road Conditions')
        c1, c2 = st.columns(2)
        with c1:
            temperature  = st.slider('Temperature (°F)',      -30.0, 130.0, 65.0)
            wind_chill   = st.slider('Wind Chill (°F)',        -50.0, 130.0, 60.0)
            humidity     = st.slider('Humidity (%)',             0,   100,   60)
            pressure     = st.slider('Pressure (in)',           25.0,  35.0, 29.9)
            visibility   = st.slider('Visibility (mi)',          0.0,  10.0,  9.0)
        with c2:
            wind_speed   = st.slider('Wind Speed (mph)',         0.0,  80.0, 10.0)
            precipitation= st.slider('Precipitation (in)',       0.0,   5.0,  0.0)
            distance     = st.slider('Distance (mi)',            0.0,  20.0,  0.3)
            start_lat    = st.slider('Latitude',                24.0,  50.0, 37.5)
            start_lng    = st.slider('Longitude',             -125.0, -65.0,-96.0)

        # ── Categorical Features (3) ──────────────────────────────
        st.markdown('####  Categorical Features')
        c3, c4, c5 = st.columns(3)
        with c3:
            weather_condition = st.number_input('Weather Condition (encoded)', min_value=0, max_value=200, value=0)
        with c4:
            wind_direction    = st.number_input('Wind Direction (encoded)',    min_value=0, max_value=30,  value=0)
        with c5:
            state             = st.number_input('State (encoded)',             min_value=0, max_value=60,  value=5)

        # ── Boolean Road Features (13) ────────────────────────────
        st.markdown('####  Road Features')
        b1, b2, b3, b4, b5 = st.columns(5)
        amenity         = int(b1.checkbox('Amenity'))
        bump            = int(b2.checkbox('Bump'))
        crossing        = int(b3.checkbox('Crossing'))
        give_way        = int(b4.checkbox('Give Way'))
        junction        = int(b5.checkbox('Junction'))
        b6, b7, b8, b9, b10 = st.columns(5)
        no_exit         = int(b6.checkbox('No Exit'))
        railway         = int(b7.checkbox('Railway'))
        roundabout      = int(b8.checkbox('Roundabout'))
        station         = int(b9.checkbox('Station'))
        stop            = int(b10.checkbox('Stop'))
        b11, b12, b13, _, _ = st.columns(5)
        traffic_calming = int(b11.checkbox('Traffic Calming'))
        traffic_signal  = int(b12.checkbox('Traffic Signal'))
        turning_loop    = int(b13.checkbox('Turning Loop'))

        # ── Temporal Features (6) ─────────────────────────────────
        st.markdown('#### Time Features')
        c6, c7 = st.columns(2)
        with c6:
            hour        = st.slider('Hour of Day',         0,  23, 8)
            day_of_week = st.slider('Day of Week (0=Mon)', 0,   6, 0)
            month       = st.slider('Month',               1,  12, 6)
        with c7:
            is_weekend  = int(day_of_week >= 5)
            is_rush     = int(hour in [7, 8, 9, 17, 18, 19])
            is_night    = int(hour < 6 or hour >= 20)
            st.info(f'Auto-computed:\n\n'
                    f'IsWeekend: **{is_weekend}** | IsRushHour: **{is_rush}** | IsNight: **{is_night}**')

      
        features = np.array([[
            temperature, wind_chill, humidity, pressure, visibility,
            wind_speed, precipitation, distance, start_lat, start_lng,
            weather_condition, wind_direction, state,
            amenity, bump, crossing, give_way, junction,
            no_exit, railway, roundabout, station, stop,
            traffic_calming, traffic_signal, turning_loop,
            hour, day_of_week, month, is_weekend, is_rush, is_night
        ]])  # shape → (1, 32)

        if st.button('🔍 Predict Severity', type='primary'):
            feat_scaled = scaler.transform(features)
            pred        = best_model.predict(feat_scaled)[0]
            prob        = best_model.predict_proba(feat_scaled)[0].max()
            label = 'High Severity ' if pred == 1 else 'Low Severity '
            color = '#e74c3c'           if pred == 1 else '#2ecc71'
            st.markdown(f"""
            <div class='metric-box' style='border-left:4px solid {color}; margin-top:20px'>
                <div class='metric-label'>Predicted Severity</div>
                <div class='metric-value' style='color:{color}'>{label}</div>
                <div class='metric-label'>Confidence: {prob:.1%}</div>
            </div>""", unsafe_allow_html=True)

    # Severity Map
    st.markdown('---')
    st.subheader('🗺️ Severity Map')
    import streamlit.components.v1 as components
    path = SAVE_DIR + 'severity_map.html'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=500, scrolling=True)
    else:
        st.warning('Severity map not found — run notebook first.')

    st.markdown(FOOTER, unsafe_allow_html=True)

# ================================================================
# PAGE 5 — HOTSPOT CLUSTERS
# ================================================================
elif page == 'Hotspot Clusters':
    st.title(' K-Means Hotspot Clusters')
    st.markdown('---')

    if not artefacts_ok:
        st.warning('Run the full notebook first to load cluster data.')
    else:
        st.subheader('Cluster Summary Table')
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('Accidents per Cluster')
        fig, ax = plt.subplots(figsize=(9, 4), facecolor='#1e2130')
        dark_ax(ax)
        cids   = cluster_summary.iloc[:, 0].astype(str)
        counts = cluster_summary['count'] if 'count' in cluster_summary.columns else cluster_summary.iloc[:, 1]
        pal = ['#e74c3c','#e67e22','#f1c40f','#2ecc71',
               '#1abc9c','#3498db','#9b59b6','#e91e63']
        ax.bar(cids, counts, color=pal[:len(cids)])
        ax.set_xlabel('Cluster ID'); ax.set_ylabel('Number of Accidents')
        ax.set_title('Cluster Size Distribution', color='white')
        st.pyplot(fig)

    st.markdown('---')

    import streamlit.components.v1 as components

    for label, fname in [
        (' Accident Heatmap',    'heatmap_accidents.html'),
        (' Cluster Hotspot Map', 'cluster_hotspots.html'),
    ]:
        path = SAVE_DIR + fname
        st.subheader(label)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=500, scrolling=True)
        else:
            st.warning(f'{label} not found — run notebook first.')
        st.markdown('---')

    st.markdown(FOOTER, unsafe_allow_html=True)

