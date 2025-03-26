import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
import ast
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns  # (optioneel, als je geen seaborn wilt gebruiken, kun je het eruit laten)
from folium.plugins import HeatMapWithTime
from branca.element import Template, MacroElement

# === 1. TITEL VAN JE APP ===
st.title("Vluchtdata uit Sensornet API")

# === 2. DATA INLADEN & VOORBEWERKING ===
default_start = '2025-03-01'
default_end = '2025-03-08'

@st.cache_data(show_spinner="📡 Data ophalen van Sensornet API...")
def laad_sensornet_data(start_date, end_date):
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts = int(pd.to_datetime(end_date).timestamp())

    url = f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_ts}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_ts}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags'
    response = requests.get(url)
    colnames = pd.DataFrame(response.json()['metadata'])
    data = pd.DataFrame(response.json()['rows'])
    data.columns = colnames.headers
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Data laden
data = laad_sensornet_data(default_start, default_end)
data['time'] = pd.to_datetime(data['time'])

keuze = st.sidebar.selectbox(
    "Kies een onderdeel",
    [
        "Dataoverzicht",
        "Heatmap geluid (per uur)",
        "Geluidsvergelijking per vliegtuigtype",
        "Hoogte vs geluid (regressie)",
        "Grootteklasse vs geluid",
        "Daggemiddelde geluid",
        "Windrichting vs geluid"
    ]
)

# === 4. GEMEENSCHAPPELIJKE FILTERS: DATUM & LOCATIES ===
# (je kunt er ook voor kiezen filters per pagina te zetten als die erg verschillend zijn)

st.sidebar.subheader("Filters")

start_date = st.sidebar.date_input("Startdatum", data['time'].min().date())
end_date = st.sidebar.date_input("Einddatum", data['time'].max().date())

# Filter de data
df = data[(data['time'].dt.date >= start_date) & (data['time'].dt.date <= end_date)]

# Locatiefilter (optioneel)
if 'location_long' in df.columns:
    locaties = df['location_long'].dropna().unique().tolist()
    geselecteerde_locaties = st.sidebar.multiselect("Locatie(s) selecteren", locaties, default=locaties)
    df = df[df['location_long'].isin(geselecteerde_locaties)]

# Voor eventuele lat/lon-parsing (nodig voor de heatmap):
def parse_location(loc):
    try:
        if isinstance(loc, str):
            return ast.literal_eval(loc)
        elif isinstance(loc, list):
            return loc
    except:
        return [None, None]

df[['lat', 'lon']] = df['location_long'].apply(parse_location).apply(pd.Series)

# === 5. PAGINA: DATAOVERZICHT ===
if keuze == "Dataoverzicht":
    st.header("🔍 Algemene informatie en data")
    st.markdown(f"Aantal rijen: **{len(df)}** | Aantal kolommen: **{df.shape[1]}**")
    st.markdown("**Kolommen:**")
    st.json(df.columns.to_list())

    st.subheader("📋 Dataoverzicht (eerste 100 rijen)")
    st.dataframe(df.head(100))

    st.write("📅 Periode in deze subset:", df['time'].min(), "tot", df['time'].max())


# === 6. PAGINA: HEATMAP GELUID (PER UUR) ===
elif keuze == "Heatmap geluid (per uur)":
    st.header("🔥 Geluidsheatmap per tijdstip")

    # Sensorlocaties: precompute aparte dicts voor latitude en longitude
    sensor_coords = {
        'Aa': [52.263, 4.750],
        'Bl': [52.271, 4.785],
        'Cn': [52.290, 4.725],
        'Ui': [52.245, 4.770],
        'Ho': [52.287, 4.780],
        'Da': [52.310, 4.740],
        'Ku': [52.275, 4.760],
        'Co': [52.265, 4.730],
    }
    sensor_lat = {k: v[0] for k, v in sensor_coords.items()}
    sensor_lon = {k: v[1] for k, v in sensor_coords.items()}

    df['lat'] = df['location_short'].map(sensor_lat)
    df['lon'] = df['location_short'].map(sensor_lon)
    df['hour'] = df['time'].dt.hour

    geselecteerd_uur = st.slider("🕒 Kies een uur", 0, 23, 12)
    filtered = df[df['hour'] == geselecteerd_uur].dropna(subset=['lat', 'lon', 'SEL_dB'])

    # Statistieken
    if not filtered.empty:
        min_val = round(filtered['SEL_dB'].min(), 1)
        max_val = round(filtered['SEL_dB'].max(), 1)
    else:
        min_val = max_val = "-"
    st.markdown(f"""
    ### 🔎 Geluidsmetingen om {geselecteerd_uur}:00 uur  
    • Aantal meetpunten: **{len(filtered)}**  
    • Sound Exposure Level in decibel: **{min_val} dB** tot **{max_val} dB**
    """)

    # Automatische centrering
    center = [filtered['lat'].mean(), filtered['lon'].mean()] if not filtered.empty else [52.3, 4.75]

    import colorsys
    def dB_naar_kleur(sel_db, min_dB=30, max_dB=90):
        norm = min(max((sel_db - min_dB) / (max_dB - min_dB), 0), 1)
        hue = (1 - norm) * 0.4
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 1)]
        return f'rgb({r},{g},{b})'

    m = folium.Map(location=center, zoom_start=12)

    # Gebruik itertuples voor een snellere iteratie
    for row in filtered.itertuples(index=False):
        kleur = dB_naar_kleur(row.SEL_dB, min_dB=30, max_dB=90)
        radius = max(5, row.SEL_dB / 3)
        popup_html = f"""
        📍 <b>{row.location_short}</b><br>
        🕒 {row.time.strftime('%H:%M')}<br>
        🖊 <b>{round(row.SEL_dB,1)} dB</b>
        """
        tooltip_text = f"{row.location_short} – {round(row.SEL_dB, 1)} dB"

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=radius,
            color=kleur,
            fill=True,
            fill_color=kleur,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=folium.Tooltip(tooltip_text, sticky=True)
        ).add_to(m)

    # Vluchtdata laden met caching
    @st.cache_data
    def load_flight_data():
        flight_df = pd.read_csv("flights_today_master1.zip", compression='zip')
        flight_df['ParsedTime'] = pd.to_datetime(flight_df['Time'], format='%a %I:%M:%S %p', errors='coerce')
        flight_df['hour'] = flight_df['ParsedTime'].dt.hour
        return flight_df.dropna(subset=['Latitude', 'Longitude', 'hour'])

    flight_df = load_flight_data()

    # Filter vluchtdata op het geselecteerde uur en groepeer per vlucht
    flight_hour_df = flight_df[flight_df['hour'] == geselecteerd_uur]
    for flight, group in flight_hour_df.groupby('FlightNumber'):
        coords = group[['Latitude', 'Longitude']].values.tolist()
        if len(coords) >= 2:
            folium.PolyLine(
                coords,
                color="blue",
                weight=2,
                opacity=0.6,
                tooltip=f"Vlucht {flight}"
            ).add_to(m)

    # Voeg de legenda maar één keer toe
    legend_html = """
    {% macro html() %}
    <div style='
        position: fixed; 
        bottom: 30px; left: 30px; width: 220px;
        background-color: white;
        padding: 10px;
        border: 2px solid gray;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        font-size: 14px;
        z-index: 9999;
    '>
        <b>🗀 Legenda</b><br>
        <i style="color:green;">●</i> Stil geluid<br>
        <i style="color:orange;">●</i> Gemiddeld geluid<br>
        <i style="color:red;">●</i> Hoog geluid<br>
        <i style="color:blue;">━</i> Vliegtuigroute
    </div>
    {% endmacro %}
    """
    from branca.element import Template, MacroElement
    legenda = MacroElement()
    legenda._template = Template(legend_html)
    m.get_root().add_child(legenda)

    st_folium(m, width=750, height=500)


# === 7. PAGINA: GELUIDSVERGELIJKING PER VLIEGTUIGTYPE (capacititeiten) ===
elif keuze == "Geluidsvergelijking per vliegtuigtype":
    st.header("📊 Geluidsvergelijking per vliegtuigtype")

# === NIEUWE PAGINA: DAGGEMIDDELDE GELUID (INTERACTIEF MET PLOTLY) ===
elif keuze == "Daggemiddelde geluid":
    st.header("📅 Gemiddeld geluidsniveau per dag (interactief)")

    import plotly.express as px  # ← zet bovenaan als je wilt

    if 'time' not in df.columns or 'SEL_dB' not in df.columns:
        st.warning("Vereiste kolommen niet beschikbaar.")
    else:
        df_dag = df.dropna(subset=["time", "SEL_dB"]).copy()
        df_dag["datum"] = df_dag["time"].dt.date

        gemiddeld_per_dag = df_dag.groupby("datum")["SEL_dB"].mean().reset_index()

        fig = px.line(
            gemiddeld_per_dag,
            x="datum",
            y="SEL_dB",
            markers=True,
            title="📅 Dagelijkse trend in geluidsniveau (SEL_dB)",
            labels={"datum": "Datum", "SEL_dB": "Gemiddeld geluidsniveau (dB)"},
        )
        fig.update_traces(line=dict(color='mediumturquoise', width=3), marker=dict(size=8))
        fig.update_layout(
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            template="simple_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        📊 Deze interactieve grafiek laat het **gemiddelde geluidsniveau (SEL_dB)** per dag zien:

        - Hover over een punt voor exacte waarden  
        - 🔍 In- en uitzoomen mogelijk  
        - 🗓️ Helpt om trends in geluidsbelasting per dag te zien
        """)

# === NIEUWE PAGINA: WINDRICHTING VS GELUID ===
elif keuze == "Windrichting vs geluid":
    st.header("🧭 Windrichting vs geluid")

    df_wind = df.dropna(subset=["winddirection", "SEL_dB"]).copy()

    # 📌 Uitleg boven de grafiek
    st.markdown("""
    ### 📌 Wat zie je hier?
    Deze windroos toont het **gemiddeld geluidsniveau (SEL_dB)** per windhoek:

    - Hoe dichter bij de rand, hoe harder het geluid
    - Richting = waar de wind vandaan komt (0° = noord, 180° = zuid)
    - Wind uit bepaalde richtingen (bijv. ZW of NO) kan leiden tot **meer geluidsoverlast**
    """)

    # Afronden op 10 graden
    df_wind["wind_rounded"] = (df_wind["winddirection"] // 10 * 10).astype(int)
    polar_df = df_wind.groupby("wind_rounded")["SEL_dB"].mean().reset_index()
    polar_df["wind_rounded_str"] = polar_df["wind_rounded"].astype(str) + "°"
    polar_df["label"] = polar_df["SEL_dB"].round(1).astype(str) + " dB"

    import plotly.express as px

    fig_polar = px.line_polar(
        polar_df,
        r="SEL_dB",
        theta="wind_rounded_str",
        line_close=True,
        markers=True,
        title="🌀 Windroos: Gemiddeld SEL_dB per windhoek",
        hover_data={"label": True}
    )

    fig_polar.update_traces(
        fill='toself',
        opacity=0.6,
        line=dict(color="mediumturquoise", width=3),
        marker=dict(color="teal", size=7)
    )

    fig_polar.update_layout(
    polar=dict(
        radialaxis=dict(
            range=[0, 100],  # <--- fix hier!
            showticklabels=True,
            ticks='',
            linewidth=1,
            gridcolor="lightgray"
        ),
        angularaxis=dict(direction="clockwise", rotation=90)
    ),
    showlegend=False,
    template="simple_white"
)


    st.plotly_chart(fig_polar, use_container_width=True)
