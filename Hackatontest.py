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
        "Stijgend vs dalend",
        "Daggemiddelde geluid",
        "Windrichting vs geluid",
        "Geluid vs Gewicht per vliegtuigtype"  # 👈 toegevoegd
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

    # Sensorlocaties hardcoded fallback
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

    df['lat'] = df['location_short'].map(lambda x: sensor_coords.get(x, [None, None])[0])
    df['lon'] = df['location_short'].map(lambda x: sensor_coords.get(x, [None, None])[1])
    df['hour'] = df['time'].dt.hour

    geselecteerd_uur = st.slider("🕒 Kies een uur", 0, 23, 12)
    filtered = df[(df['hour'] == geselecteerd_uur)].dropna(subset=['lat', 'lon', 'SEL_dB'])

    # 🧐 Stat info
    min_val = round(filtered['SEL_dB'].min(), 1) if not filtered.empty else "-"
    max_val = round(filtered['SEL_dB'].max(), 1) if not filtered.empty else "-"
    st.markdown(f"""
    ### 🔎 Geluidsmetingen om {geselecteerd_uur}:00 uur  
    • Aantal meetpunten: **{len(filtered)}**  
    • Sound Exposure Level in decibel: **{min_val} dB** tot **{max_val} dB**
    """)

    # Automatische centrering
    if not filtered.empty:
        center = [filtered['lat'].mean(), filtered['lon'].mean()]
    else:
        center = [52.3, 4.75]

    import colorsys
    def dB_naar_kleur(sel_db, min_dB=30, max_dB=90):
        norm = min(max((sel_db - min_dB) / (max_dB - min_dB), 0), 1)
        hue = (1 - norm) * 0.4
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 1)]
        return f'rgb({r},{g},{b})'

    m = folium.Map(location=center, zoom_start=12)

    for _, row in filtered.iterrows():
        kleur = dB_naar_kleur(row["SEL_dB"], min_dB=30, max_dB=90)
        radius = max(5, row["SEL_dB"] / 3)
        popup_html = f"""
        📍 <b>{row['location_short']}</b><br>
        🕒 {row['time'].strftime('%H:%M')}<br>
        🖊 <b>{round(row['SEL_dB'],1)} dB</b>
        """
        tooltip_text = f"{row['location_short']} – {round(row['SEL_dB'], 1)} dB"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=kleur,
            fill=True,
            fill_color=kleur,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=folium.Tooltip(tooltip_text, sticky=True)
        ).add_to(m)

# === Vluchtdata toevoegen ===
# Cache als globale variabele (eenmalig inlezen)
if 'cached_flight_df' not in globals():
    print("CSV inlezen en verwerken...")
    flight_df = pd.read_csv("flights_today_master1.zip", compression='zip')
    flight_df['ParsedTime'] = pd.to_datetime(flight_df['Time'], format='%a %I:%M:%S %p', errors='coerce')
    flight_df['hour'] = flight_df['ParsedTime'].dt.hour
    flight_df = flight_df.dropna(subset=['Latitude', 'Longitude', 'hour'])
    
    # Cache het resultaat
    cached_flight_df = flight_df
else:
    flight_df = cached_flight_df

# Groeperen
grouped_flights = flight_df.groupby(['FlightNumber', 'hour'])

for (vlucht, uur), groep in grouped_flights:
    if uur != geselecteerd_uur:
        continue
    coords = groep[['Latitude', 'Longitude']].values.tolist()
    if len(coords) >= 2:
        folium.PolyLine(
            coords,
            color="blue",
            weight=2,
            opacity=0.6,
            tooltip=f"Vlucht {vlucht}"
        ).add_to(m)


    # 📘 Legenda: linksonder + nettere titel
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

    data_clean = df.dropna(subset=["icao_type", "SEL_dB"])

    # Capaciteit + grootteklasse
    capaciteit_data = {
        "A320": {"passagiers": 180, "vracht_ton": 2.5},
        "A319": {"passagiers": 140, "vracht_ton": 2.0},
        "A321": {"passagiers": 220, "vracht_ton": 3.0},
        "B738": {"passagiers": 189, "vracht_ton": 2.6},
        "B737": {"passagiers": 162, "vracht_ton": 2.4},
        "B744": {"passagiers": 416, "vracht_ton": 20.0},
        "B77W": {"passagiers": 396, "vracht_ton": 23.0},
        "B77F": {"passagiers": 0,   "vracht_ton": 112.0},
        "A332": {"passagiers": 278, "vracht_ton": 15.0},
        "A333": {"passagiers": 277, "vracht_ton": 16.0},
        "E190": {"passagiers": 100, "vracht_ton": 1.5},
        "CRJ2": {"passagiers": 50,  "vracht_ton": 0.8},
    }

    grootteklasse = {
        "CRJ2": "Klein", "E190": "Klein",
        "A319": "Middelgroot", "A320": "Middelgroot", "A321": "Middelgroot",
        "B737": "Middelgroot", "B738": "Middelgroot",
        "A332": "Groot", "A333": "Groot", "B744": "Groot", "B77W": "Groot", "B77F": "Groot"
    }

    capaciteit_df = pd.DataFrame(capaciteit_data).T.reset_index()
    capaciteit_df.columns = ["icao_type", "passagiers", "vracht_ton"]
    capaciteit_df["grootteklasse"] = capaciteit_df["icao_type"].map(grootteklasse)

    gemiddeld_geluid = data_clean.groupby("icao_type")["SEL_dB"].mean().reset_index()
    gemiddeld_geluid.columns = ["icao_type", "gemiddeld_SEL_dB"]

    resultaat = pd.merge(gemiddeld_geluid, capaciteit_df, on="icao_type", how="left")
    resultaat["geluid_per_passagier"] = resultaat["gemiddeld_SEL_dB"] / resultaat["passagiers"]
    resultaat["geluid_per_ton_vracht"] = resultaat["gemiddeld_SEL_dB"] / resultaat["vracht_ton"]

    # Filter out infs
    resultaat = resultaat.replace([float("inf"), -float("inf")], np.nan)

    # Selectie
    keuze_metric = st.radio("Kies wat je wilt vergelijken:", ["Per passagier", "Per ton vracht"])
    if keuze_metric == "Per passagier":
        kolom = "geluid_per_passagier"
        titel = "Geluidsbelasting per passagier"
    else:
        kolom = "geluid_per_ton_vracht"
        titel = "Geluidsbelasting per ton vracht"

    resultaat_plot = resultaat.dropna(subset=[kolom]).sort_values(by=kolom).reset_index(drop=True)

    # Plotly kleuren
    import plotly.express as px

    kleurmap_plotly = {
        "Klein": "#FFD700",
        "Middelgroot": "#FF7F0E",
        "Groot": "#D62728"
    }

    fig = px.bar(
        resultaat_plot,
        x=kolom,
        y="icao_type",
        orientation="h",
        color="grootteklasse",
        color_discrete_map=kleurmap_plotly,
        labels={
            kolom: "Gemiddelde geluidsbelasting (dB)",
            "icao_type": "Vliegtuigtype",
            "grootteklasse": "Grootteklasse"
        },
        hover_data=["gemiddeld_SEL_dB", "passagiers", "vracht_ton"]
    )
    fig.update_layout(
        title=f"🎯 Interactieve vergelijking ({kolom})",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

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
elif keuze == "Grootteklasse vs geluid":
    st.header("✈️ Gewicht vs geluid per vliegtuigtype (visueel verbeterd)")

    # === Data voorbereiden ===
    gewicht_per_type = {
        "CRJ2": 24000, "E190": 47000, "A319": 64000, "A320": 73500, "A321": 89000,
        "B738": 79000, "B737": 71000, "A332": 175000, "A333": 195000,
        "B744": 396000, "B77W": 351000, "B77F": 347800
    }

    grootteklasse = {
        "CRJ2": "Klein", "E190": "Klein", "A319": "Middelgroot", "A320": "Middelgroot", "A321": "Middelgroot", 
        "B738": "Middelgroot", "B737": "Middelgroot", "A332": "Groot", "A333": "Groot", 
        "B744": "Groot", "B77W": "Groot", "B77F": "Groot"
    }

    gewicht_df = pd.DataFrame.from_dict(gewicht_per_type, orient="index", columns=["gewicht_kg"]).reset_index()
    gewicht_df = gewicht_df.rename(columns={"index": "icao_type"})
    gewicht_df["grootteklasse"] = gewicht_df["icao_type"].map(grootteklasse)

    sel_df = df.dropna(subset=["icao_type", "SEL_dB"]).groupby("icao_type")["SEL_dB"].mean().reset_index()
    sel_df.columns = ["icao_type", "gemiddeld_SEL_dB"]

    plot_df = gewicht_df.merge(sel_df, on="icao_type").dropna()

    # === Scatterplot met trendlijn ===
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definieer de gewenste volgorde van de grootteklassen
    hue_order = ["Klein", "Middelgroot", "Groot"]

    # Maak de scatterplot, maar schakel de automatische legende uit
    scatter = sns.scatterplot(
        data=plot_df,
        x="gewicht_kg",
        y="gemiddeld_SEL_dB",
        hue="grootteklasse",
        hue_order=hue_order,
        size="gewicht_kg",
        palette="Set2",
        sizes=(60, 300),
        ax=ax,
        legend=False  # automatisch genereren van de legende uitschakelen
    )

    # Voeg de regressielijn toe zoals voorheen
    sns.regplot(
        data=plot_df,
        x="gewicht_kg",
        y="gemiddeld_SEL_dB",
        scatter=False,
        ax=ax,
        color="red",
        line_kws={"linestyle": "dashed", "linewidth": 2}
    )

    # Labels per datapunt toevoegen
    for _, row in plot_df.iterrows():
        ax.text(row["gewicht_kg"], row["gemiddeld_SEL_dB"] + 0.3, row["icao_type"],
                ha="center", fontsize=9)

    ax.set_title("✈️ Gewicht vs gemiddeld geluid per vliegtuigtype", fontsize=14)
    ax.set_xlabel("Gemiddeld gewicht vliegtuigtype (kg)", fontsize=12)
    ax.set_ylabel("Gemiddeld SEL_dB", fontsize=12)

    # Maak een aangepaste legende voor de kleurcategorieën
    from matplotlib.lines import Line2D
    # Haal de kleuren op uit de gebruikte palette
    palette_colors = sns.color_palette("Set2", n_colors=len(hue_order))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=cat,
            markerfacecolor=palette_colors[i], markersize=10)
        for i, cat in enumerate(hue_order)
    ]
    ax.legend(handles=legend_elements, title="Grootteklasse", loc="lower right", frameon=True)
    plt.tight_layout()
    st.pyplot(fig)


    # === Uitleg onder de grafiek ===
    st.markdown("""
    🔍 Deze grafiek toont hoe het **gewicht van het vliegtuigtype** samenhangt met het **gemiddeld geluidsniveau**.

    - ⚫ Grote cirkels = zwaarder toestel  
    - 📉 De rode stippellijn toont de gemiddelde trend: **zwaarder = luider**  
    - ✈️ Grote toestellen zoals **B744** en **B77W** hebben duidelijk meer gewicht én geluid  
    - 💡 Dit bevestigt dat **zwaardere vliegtuigen vaker zorgen voor meer geluidsoverlast**
    """)


# === 11. PAGINA: STIJGEND VS DALEND ===
elif keuze == "Stijgend vs dalend":
    st.header("🛫✈️ Stijgen vs. Landen: geluidsverdeling per richting")

    def bepaal_richting(altitude, distance):
        if pd.isna(altitude) or pd.isna(distance):
            return "Onbekend"
        if altitude > 1000 and distance < 20:
            return "Stijgend"
        elif altitude < 800 and distance > 20:
            return "Dalend"
        else:
            return "Onbekend"

    # Voeg toe aan df als nog niet bestaat
    if "vluchtrichting" not in df.columns:
        df["vluchtrichting"] = df.apply(
            lambda row: bepaal_richting(row["altitude"], row["distance"]), axis=1
        )

    df_richting = df[df["vluchtrichting"].isin(["Stijgend", "Dalend"])].dropna(subset=["SEL_dB"])
    df_stijgend = df_richting[df_richting["vluchtrichting"] == "Stijgend"]
    df_dalend = df_richting[df_richting["vluchtrichting"] == "Dalend"]

    aantal_stijgend = len(df_stijgend)
    aantal_dalend = len(df_dalend)

    st.markdown(f"""
    📊 **Aantal herkende vluchten**
    - 🛫 Stijgend: **{aantal_stijgend}**
    - 🛬 Dalend: **{aantal_dalend}**
    """)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Stijgend
    if aantal_stijgend > 0:
        sns.histplot(df_stijgend["SEL_dB"], bins=20, kde=True, ax=axes[0])
        axes[0].set_title("🛫 Stijgend", fontsize=13)
    else:
        axes[0].text(0.5, 0.5, "Geen data", ha="center", va="center", fontsize=12)
        axes[0].set_title("🛫 Stijgend (geen data)", fontsize=13)
    axes[0].set_xlabel("SEL_dB")
    axes[0].set_ylabel("Aantal metingen")

    # Dalend
    if aantal_dalend > 0:
        sns.histplot(df_dalend["SEL_dB"], bins=20, kde=True, ax=axes[1])
        axes[1].set_title("🛬 Dalend", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "Geen data", ha="center", va="center", fontsize=12)
        axes[1].set_title("🛬 Dalend (geen data)", fontsize=13)
    axes[1].set_xlabel("SEL_dB")

    fig.suptitle("🔊 Geluidsverdeling bij stijgende en dalende vluchten", fontsize=15)
    st.pyplot(fig)

    st.markdown("""
    - Links: **Stijgende vluchten**  
    - Rechts: **Dalende vluchten**  

    💡 Meestal geeft opstijgen (stijgend) meer geluidsoverlast.
    """)


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

####
elif keuze == "Geluid vs Gewicht per vliegtuigtype":
    st.header("⚖️ Geluid vs Gewicht per vliegtuigtype")
