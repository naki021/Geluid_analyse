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

@st.cache_data(show_spinner="Data ophalen van Sensornet API...")
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

# Dynamisch ophalen van unieke vliegtuigtypes uit de dataset
unieke_types = data['icao_type'].dropna().unique()

# Mapping van vliegtuigtypes naar gewichtsklasse, MTOW, passagierscapaciteit en vrachtcapaciteit (ton)
gewicht_info = {
    'CRJ2': {'klasse': 'middelzwaar', 'gewicht_kg': 24000, 'passagiers': 50, 'vracht_ton': 1.2},
    'A333': {'klasse': 'zwaar', 'gewicht_kg': 242000, 'passagiers': 277, 'vracht_ton': 16.0},
    'A310': {'klasse': 'zwaar', 'gewicht_kg': 164000, 'passagiers': 280, 'vracht_ton': 15.0},
    'E75L': {'klasse': 'middelzwaar', 'gewicht_kg': 38500, 'passagiers': 88, 'vracht_ton': 1.8},
    'A332': {'klasse': 'zwaar', 'gewicht_kg': 242000, 'passagiers': 278, 'vracht_ton': 15.0},
    'A359': {'klasse': 'zwaar', 'gewicht_kg': 280000, 'passagiers': 315, 'vracht_ton': 19.0},
    'B77W': {'klasse': 'zwaar', 'gewicht_kg': 351500, 'passagiers': 396, 'vracht_ton': 23.0},
    'B789': {'klasse': 'zwaar', 'gewicht_kg': 254000, 'passagiers': 296, 'vracht_ton': 19.0},
    'B738': {'klasse': 'middelzwaar', 'gewicht_kg': 79015, 'passagiers': 189, 'vracht_ton': 2.6},
    'E190': {'klasse': 'middelzwaar', 'gewicht_kg': 51000, 'passagiers': 100, 'vracht_ton': 1.5},
    'B737': {'klasse': 'middelzwaar', 'gewicht_kg': 70500, 'passagiers': 162, 'vracht_ton': 2.4},
    'B739': {'klasse': 'middelzwaar', 'gewicht_kg': 79000, 'passagiers': 215, 'vracht_ton': 2.8},
    'B744': {'klasse': 'zwaar', 'gewicht_kg': 396000, 'passagiers': 416, 'vracht_ton': 20.0},
    'A320': {'klasse': 'middelzwaar', 'gewicht_kg': 77000, 'passagiers': 180, 'vracht_ton': 2.5},
    'BCS3': {'klasse': 'middelzwaar', 'gewicht_kg': 62700, 'passagiers': 145, 'vracht_ton': 2.2},
    'A319': {'klasse': 'middelzwaar', 'gewicht_kg': 64000, 'passagiers': 140, 'vracht_ton': 2.0},
    'B77L': {'klasse': 'zwaar', 'gewicht_kg': 347800, 'passagiers': 0, 'vracht_ton': 112.0},
    'B772': {'klasse': 'zwaar', 'gewicht_kg': 297500, 'passagiers': 375, 'vracht_ton': 18.5},
    'C25B': {'klasse': 'licht', 'gewicht_kg': 5670, 'passagiers': 8, 'vracht_ton': 0.5},
    'E55P': {'klasse': 'licht', 'gewicht_kg': 8355, 'passagiers': 8, 'vracht_ton': 0.6},
    'C680': {'klasse': 'licht', 'gewicht_kg': 9900, 'passagiers': 12, 'vracht_ton': 0.7},
    'C68A': {'klasse': 'licht', 'gewicht_kg': 9900, 'passagiers': 12, 'vracht_ton': 0.7},
    'E170': {'klasse': 'middelzwaar', 'gewicht_kg': 38500, 'passagiers': 76, 'vracht_ton': 1.6},
    'C56X': {'klasse': 'licht', 'gewicht_kg': 9500, 'passagiers': 8, 'vracht_ton': 0.6},
    'CRJ9': {'klasse': 'middelzwaar', 'gewicht_kg': 34000, 'passagiers': 90, 'vracht_ton': 1.5},
    'F900': {'klasse': 'middelzwaar', 'gewicht_kg': 22000, 'passagiers': 19, 'vracht_ton': 1.3},
    'C25A': {'klasse': 'licht', 'gewicht_kg': 5670, 'passagiers': 7, 'vracht_ton': 0.4},
    'B773': {'klasse': 'zwaar', 'gewicht_kg': 351500, 'passagiers': 368, 'vracht_ton': 22.0},
    'B788': {'klasse': 'zwaar', 'gewicht_kg': 227900, 'passagiers': 242, 'vracht_ton': 17.0},
    'C525': {'klasse': 'licht', 'gewicht_kg': 5670, 'passagiers': 6, 'vracht_ton': 0.3},
    'A388': {'klasse': 'zwaar', 'gewicht_kg': 575000, 'passagiers': 853, 'vracht_ton': 38.0},
    'A20N': {'klasse': 'middelzwaar', 'gewicht_kg': 79000, 'passagiers': 195, 'vracht_ton': 2.8},
    'EC35': {'klasse': 'licht', 'gewicht_kg': 2980, 'passagiers': 6, 'vracht_ton': 0.1},
    'B38M': {'klasse': 'middelzwaar', 'gewicht_kg': 82000, 'passagiers': 210, 'vracht_ton': 3.0},
    'A321': {'klasse': 'middelzwaar', 'gewicht_kg': 93000, 'passagiers': 220, 'vracht_ton': 3.0},
    'B78X': {'klasse': 'zwaar', 'gewicht_kg': 254000, 'passagiers': 318, 'vracht_ton': 18.5},
    'A306': {'klasse': 'zwaar', 'gewicht_kg': 155000, 'passagiers': 266, 'vracht_ton': 13.5},
    'A318': {'klasse': 'middelzwaar', 'gewicht_kg': 68000, 'passagiers': 132, 'vracht_ton': 1.9},
    'B748': {'klasse': 'zwaar', 'gewicht_kg': 448000, 'passagiers': 467, 'vracht_ton': 34.0},
    'C510': {'klasse': 'licht', 'gewicht_kg': 4600, 'passagiers': 4, 'vracht_ton': 0.3},
    'F70': {'klasse': 'middelzwaar', 'gewicht_kg': 41000, 'passagiers': 80, 'vracht_ton': 1.4},
    'CL35': {'klasse': 'licht', 'gewicht_kg': 8618, 'passagiers': 10, 'vracht_ton': 0.6},
    'E35L': {'klasse': 'middelzwaar', 'gewicht_kg': 22500, 'passagiers': 13, 'vracht_ton': 1.0},
    'A21N': {'klasse': 'middelzwaar', 'gewicht_kg': 93500, 'passagiers': 230, 'vracht_ton': 3.2},
    'PC12': {'klasse': 'licht', 'gewicht_kg': 4740, 'passagiers': 9, 'vracht_ton': 0.4},
    'TBM9': {'klasse': 'licht', 'gewicht_kg': 3374, 'passagiers': 6, 'vracht_ton': 0.2},
    'F2TH': {'klasse': 'middelzwaar', 'gewicht_kg': 17960, 'passagiers': 10, 'vracht_ton': 0.8},
    'B752': {'klasse': 'middelzwaar', 'gewicht_kg': 115680, 'passagiers': 239, 'vracht_ton': 3.5},
    'GLF6': {'klasse': 'middelzwaar', 'gewicht_kg': 22600, 'passagiers': 19, 'vracht_ton': 1.0},
    'SW4': {'klasse': 'licht', 'gewicht_kg': 7300, 'passagiers': 19, 'vracht_ton': 0.4},
    'DH8D': {'klasse': 'middelzwaar', 'gewicht_kg': 29257, 'passagiers': 78, 'vracht_ton': 1.7},
    'B734': {'klasse': 'middelzwaar', 'gewicht_kg': 68000, 'passagiers': 149, 'vracht_ton': 2.2},
    'GLEX': {'klasse': 'middelzwaar', 'gewicht_kg': 22600, 'passagiers': 15, 'vracht_ton': 0.9},
    'B763': {'klasse': 'zwaar', 'gewicht_kg': 186880, 'passagiers': 260, 'vracht_ton': 15.0},
    'G650': {'klasse': 'middelzwaar', 'gewicht_kg': 45000, 'passagiers': 19, 'vracht_ton': 1.0},
    'D328': {'klasse': 'middelzwaar', 'gewicht_kg': 14600, 'passagiers': 33, 'vracht_ton': 0.8},
    'E75S': {'klasse': 'middelzwaar', 'gewicht_kg': 38500, 'passagiers': 88, 'vracht_ton': 1.8},
    'GL5T': {'klasse': 'middelzwaar', 'gewicht_kg': 22600, 'passagiers': 16, 'vracht_ton': 1.0},
    'E50P': {'klasse': 'licht', 'gewicht_kg': 5200, 'passagiers': 5, 'vracht_ton': 0.3},
    'FA7X': {'klasse': 'middelzwaar', 'gewicht_kg': 31200, 'passagiers': 12, 'vracht_ton': 0.9},
    'E135': {'klasse': 'middelzwaar', 'gewicht_kg': 20500, 'passagiers': 37, 'vracht_ton': 1.3},
    'A339': {'klasse': 'zwaar', 'gewicht_kg': 251000, 'passagiers': 296, 'vracht_ton': 18.0},
    'GLF5': {'klasse': 'middelzwaar', 'gewicht_kg': 41900, 'passagiers': 19, 'vracht_ton': 1.0},
    'B733': {'klasse': 'middelzwaar', 'gewicht_kg': 68000, 'passagiers': 148, 'vracht_ton': 2.1},
    'LJ60': {'klasse': 'licht', 'gewicht_kg': 10400, 'passagiers': 7, 'vracht_ton': 0.5},
    'PC24': {'klasse': 'licht', 'gewicht_kg': 8040, 'passagiers': 10, 'vracht_ton': 0.6},
    'A139': {'klasse': 'licht', 'gewicht_kg': 6800, 'passagiers': 15, 'vracht_ton': 0.2}
}


# Nieuwe kolommen toevoegen aan de dataframe
data['gewichtsklasse'] = data['icao_type'].map(lambda x: gewicht_info.get(x, {}).get('klasse', 'onbekend'))
data['gewicht_kg'] = data['icao_type'].map(lambda x: gewicht_info.get(x, {}).get('gewicht_kg', np.nan))


keuze = st.sidebar.selectbox(
    "Kies een onderdeel",
    [
        "Dataoverzicht",
        "Heatmap geluid (per uur)",
        "Geluidsvergelijking per vliegtuigtype",
        "Hoogte vs geluid (regressie)",
        "Gewicht vs geluid per vliegtuigtype",
        "Windrichting vs geluid",
        "Geluid vs Gewicht per grootteklasse"
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
    st.header("Algemene informatie en data")
    st.subheader("Dataoverzicht (eerste 100 rijen)")
    st.dataframe(df.head(100))

    st.write("Periode in deze subset:", df['time'].min(), "tot", df['time'].max())


# === 6. PAGINA: HEATMAP GELUID (PER UUR) ===
elif keuze == "Heatmap geluid (per uur)":
    st.header("Geluidsheatmap per tijdstip")

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

    geselecteerd_uur = st.slider("Kies een uur", 0, 23, 12)
    filtered = df[df['hour'] == geselecteerd_uur].dropna(subset=['lat', 'lon', 'SEL_dB'])

    # Statistieken
    if not filtered.empty:
        min_val = round(filtered['SEL_dB'].min(), 1)
        max_val = round(filtered['SEL_dB'].max(), 1)
    else:
        min_val = max_val = "-"
    st.markdown(f"""
    ### Geluidsmetingen om {geselecteerd_uur}:00 uur  
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
        <b>{row.location_short}</b><br>
        {row.time.strftime('%H:%M')}<br>
        <b>{round(row.SEL_dB,1)} dB</b>
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


# === 7. PAGINA: GELUIDSVERGELIJKING PER VLIEGTUIGTYPE ===
elif keuze == "Geluidsvergelijking per vliegtuigtype":
    st.header("🔊 Geluidsvergelijking per vliegtuigtype (per eenheid)")

    # Voeg passagiers- en vrachtdata toe
    data["passagiers"] = data["icao_type"].map(lambda x: gewicht_info.get(x, {}).get("passagiers", np.nan))
    data["vracht_ton"] = data["icao_type"].map(lambda x: gewicht_info.get(x, {}).get("vracht_ton", np.nan))

    # Gemiddeld geluid per vliegtuigtype
    agg_df = data.groupby("icao_type").agg(
        gemiddeld_geluid=("SEL_dB", "mean"),
        passagiers=("passagiers", "first"),
        vracht_ton=("vracht_ton", "first")
    ).reset_index()

    agg_df["geluid_per_passagier"] = agg_df["gemiddeld_geluid"] / agg_df["passagiers"]
    agg_df["geluid_per_ton_vracht"] = agg_df["gemiddeld_geluid"] / agg_df["vracht_ton"]

    # Keuze: passagier vs vracht
    optie = st.radio("Wat wil je analyseren?", ["Per passagier", "Per ton vracht"], horizontal=True)
    aantal = st.slider("Aantal vliegtuigtypes tonen", min_value=5, max_value=20, value=10)

    if optie == "Per passagier":
        kolom = "geluid_per_passagier"
        title = "🔊 Gemiddeld geluid per passagier (dB)"
        kleur = "OrRd"
        data_filter = agg_df.query("passagiers > 0").dropna(subset=[kolom])
    else:
        kolom = "geluid_per_ton_vracht"
        title = "🔊 Gemiddeld geluid per ton vracht (dB)"
        kleur = "Blues"
        data_filter = agg_df.dropna(subset=[kolom])

    top = data_filter.sort_values(by=kolom, ascending=False).head(aantal)

    if top.empty:
        st.warning("Geen vliegtuigtypes beschikbaar met geldige data voor deze vergelijking.")
    else:
        import plotly.express as px

        fig = px.bar(
            top,
            x=kolom,
            y="icao_type",
            orientation="h",
            color=kolom,
            color_continuous_scale=kleur,
            labels={
                kolom: "Geluidsbelasting (dB)",
                "icao_type": "Vliegtuigtype"
            },
            hover_data=["gemiddeld_geluid", "passagiers", "vracht_ton"]
        )
        fig.update_layout(title=title, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Wat zie je hier?**
        - Deze grafiek toont hoeveel geluid een vliegtuig gemiddeld produceert **per passagier** of **per ton vracht**.
        - Dit geeft een eerlijker beeld dan alleen kijken naar totaalgeluid.
        - Kleinere of zakelijke jets scoren vaak slechter qua efficiëntie.
        """)



# === 8. PAGINA: HOOGTE VS GELUID (REGRESSIE) ===
elif keuze == "Hoogte vs geluid (regressie)":
    st.header("✈️ Hoe hoger het vliegtuig, hoe lager het geluid")

    df_hoogte = df.dropna(subset=["SEL_dB", "altitude"])
    if len(df_hoogte) < 2:
        st.warning("Te weinig data voor een regressieplot.")
    else:
        X = df_hoogte["altitude"].values.reshape(-1, 1)
        y = df_hoogte["SEL_dB"].values

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        x0, x1 = 1000, 2000
        y0 = model.predict([[x0]])[0]
        y1 = model.predict([[x1]])[0]
        delta_y = y0 - y1

        df_hoogte["hoogte_binned"] = (df_hoogte["altitude"] // 100) * 100
        percentielen = df_hoogte.groupby("hoogte_binned")["SEL_dB"].quantile([0.1, 0.5, 0.9]).unstack()
        percentielen = percentielen.reset_index()
        percentielen.columns = ["Hoogte", "P10", "P50", "P90"]
        percentielen["Hoogte"] = percentielen["Hoogte"].astype(float)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Spreidingsband
        ax.fill_between(percentielen["Hoogte"], percentielen["P10"], percentielen["P90"],
                        alpha=0.2, color='gray', label="10–90% spreiding")
        ax.plot(percentielen["Hoogte"], percentielen["P50"], color='gray', linestyle="--", label="Mediaan")

        sns.scatterplot(data=df_hoogte, x="altitude", y="SEL_dB", alpha=0.2, ax=ax, color="orange", label="Meetpunten")
        sns.regplot(data=df_hoogte, x="altitude", y="SEL_dB", scatter=False, color="red", ax=ax, label="Gem. trend")

        ax.annotate(f"{delta_y:.1f} dB verschil", xy=((x0 + x1) / 2, (y0 + y1) / 2),
                    xytext=(x0 + 150, y0 + 1.5), arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=10, color="black")

        ax.set_title("✈️ Hoe hoger het vliegtuig, hoe lager het geluid", fontsize=14)
        ax.set_xlabel("Hoogte (m)", fontsize=12)
        ax.set_ylabel("Geluidsniveau (SEL_dB)", fontsize=12)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)

        st.markdown(f"""
        🔍 Deze grafiek toont het verband tussen **vlieghoogte** en **geluidsniveau**.
        
        - De rode lijn is de **gemiddelde trend**: geluid daalt gemiddeld **{delta_y:.1f} dB tussen 1000 en 2000 meter**.
        - De grijze band laat de spreiding per hoogte zien.
        """)


# === 9. PAGINA: BOUWJAAR VS GELUID ===
elif keuze == "Gewicht vs geluid per vliegtuigtype":
    st.header("Gewicht vs geluid per vliegtuigtype")

### TEST 3

    # Filter data
    data_clean = data.dropna(subset=["icao_type", "SEL_dB", "gewicht_kg", "gewichtsklasse"])

    # Gemiddeld geluid en gewicht per vliegtuigtype
    plot_df = data_clean.groupby("icao_type").agg({
        "SEL_dB": "mean",
        "gewicht_kg": "first",
        "gewichtsklasse": "first"
    }).reset_index()
    plot_df.columns = ["icao_type", "gemiddeld_SEL_dB", "gewicht_kg", "gewichtsklasse"]

    # Voeg kleur toe per klasse
    kleurmap = {
        "licht": "#66c2a5",
        "middelgroot": "#fc8d62",
        "zwaar": "#8da0cb"
    }

    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression
    import numpy as np

    fig = go.Figure()

    # Voeg een trace toe per gewichtsklasse
    for klasse in plot_df["gewichtsklasse"].unique():
        subset = plot_df[plot_df["gewichtsklasse"] == klasse]
        fig.add_trace(go.Scatter(
            x=subset["gewicht_kg"],
            y=subset["gemiddeld_SEL_dB"],
            mode='markers',
            marker=dict(
                size=subset["gewicht_kg"] / 8000,  # grootte op schaal
                color=kleurmap.get(klasse, "#999999"),
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.8
            ),
            name=klasse.capitalize(),
            text=subset["icao_type"],
            hovertemplate="<b>%{text}</b><br>Gewicht: %{x} kg<br>Geluidsniveau: %{y:.1f} dB<extra></extra>"
        ))

    # === Regressielijn toevoegen ===
    X = plot_df["gewicht_kg"].values.reshape(-1, 1)
    y = plot_df["gemiddeld_SEL_dB"].values
    model = LinearRegression().fit(X, y)
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line.reshape(-1, 1))

    fig.add_trace(go.Scatter(
        x=X_line,
        y=y_line,
        mode="lines",
        name="Trendlijn",
        line=dict(color="red", dash="dash"),
    ))

    # Layout
    fig.update_layout(
        title="✈️ Gewicht vs gemiddeld geluid per vliegtuigtype",
        xaxis_title="Gemiddeld gewicht vliegtuigtype (kg)",
        yaxis_title="Gemiddeld SEL (dB)",
        height=600,
        legend_title="Gewichtsklasse",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)


# === NIEUWE PAGINA: WINDRICHTING VS GELUID ===
elif keuze == "Windrichting vs geluid":
    st.header("Windrichting vs geluid")

    df_wind = df.dropna(subset=["winddirection", "SEL_dB"]).copy()

    # Uitleg boven de grafiek
    st.markdown("""
    ### Wat zie je hier?
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
        title="Windroos: Gemiddeld SEL_dB per windhoek",
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

#### plot
elif keuze == "Geluid vs Gewicht per grootteklasse":
    st.header("Geluid vs Gewicht per grootteklasse")

    # Data voorbereiden
    df_clean = df.dropna(subset=["icao_type", "SEL_dB", "time"])
    df_clean["datum"] = df_clean["time"].dt.date

    # Map elke icao_type naar de juiste grootteklasse
    grootteklasse = {
        "CRJ2": "Klein", "E190": "Klein",
        "A319": "Middelgroot", "A320": "Middelgroot", "A321": "Middelgroot", 
        "B738": "Middelgroot", "B737": "Middelgroot",
        "A332": "Groot", "A333": "Groot", "B744": "Groot", "B77W": "Groot", "B77F": "Groot"
    }
    df_clean["grootteklasse"] = df_clean["icao_type"].map(grootteklasse)

    # Groeperen per datum en grootteklasse en gemiddelde geluidsniveau berekenen
    groepsdata_grootte = df_filtered.groupby(["datum", "grootteklasse"])["SEL_dB"].mean().reset_index()

    # Plotly lijnplot: per grootteklasse over de tijd
    import plotly.express as px
    fig = px.line(
        groepsdata_grootte,
        x="datum",
        y="SEL_dB",
        color="grootteklasse",
        markers=True,
        labels={
            "SEL_dB": "Gemiddelde geluidsniveau (dB)",
            "datum": "Datum",
            "grootteklasse": "Grootteklasse"
        },
        title="Gemiddeld geluid per grootteklasse per dag"
    )

    # Y-as label verduidelijken
    fig.update_layout(
        yaxis_title="Gemiddeld geluidsniveau (SEL_dB)\n(meer = luider)"
    )

    # Visuele zones toevoegen
    min_date = groepsdata_grootte["datum"].min()
    max_date = groepsdata_grootte["datum"].max()

    fig.add_shape(type="rect", x0=min_date, x1=max_date, y0=80, y1=90,
                  fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=min_date, x1=max_date, y0=75, y1=80,
                  fillcolor="orange", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=min_date, x1=max_date, y0=70, y1=75,
                  fillcolor="lightgreen", opacity=0.1, layer="below", line_width=0)

    # Referentielijn op 75 dB
    fig.add_hline(y=75, line_dash="dot", line_color="gray",
                  annotation_text="Normgrens 75 dB", annotation_position="top left")

    st.plotly_chart(fig, use_container_width=True)
