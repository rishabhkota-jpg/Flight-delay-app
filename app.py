import json
import requests
import datetime as dt
from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Risk Checker", page_icon="✈️", layout="wide")

# ---------- look and feel ----------

ACCENT = "#e5484d"
MUTED = "#7cc4fa"
GOOD = "#30a46c"
WARN = "#e5a000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.main .block-container {
    padding-top: 2.2rem;
    max-width: 1280px;
    animation: pageIn .55s cubic-bezier(.22,1,.36,1);
}
@keyframes pageIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}

/* hero */
.hero {
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 26px;
    background:
        radial-gradient(900px 300px at 8% -10%, rgba(124,196,250,.20), transparent 60%),
        radial-gradient(700px 300px at 95% 0%, rgba(229,72,77,.16), transparent 60%),
        rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.09);
    animation: pageIn .7s cubic-bezier(.22,1,.36,1);
}
.hero h1 {
    margin: 0 0 .45rem 0;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -.035em;
    line-height: 1.1;
}
.hero .plane { display: inline-block; animation: fly 4.5s ease-in-out infinite; }
@keyframes fly {
    0%,100% { transform: translate(0,0) rotate(0deg); }
    50%     { transform: translate(9px,-6px) rotate(6deg); }
}
.hero p { margin: 0; opacity: .82; font-size: 1.02rem; line-height: 1.6; max-width: 900px; }
.hero b { font-weight: 700; opacity: 1; }

/* metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform .22s cubic-bezier(.22,1,.36,1), border-color .22s, background .22s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(124,196,250,.5);
    background: rgba(255,255,255,.06);
}
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -.02em; }

/* section headings */
h2, h3 {
    letter-spacing: -.02em;
    font-weight: 700;
    padding-bottom: .35rem;
    border-bottom: 1px solid rgba(255,255,255,.07);
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    transition: background .2s, color .2s;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,.05); }

/* inputs */
div[data-baseweb="select"] > div, .stDateInput input {
    border-radius: 10px !important;
    transition: border-color .2s, box-shadow .2s;
}
div[data-baseweb="select"] > div:hover { border-color: rgba(124,196,250,.6) !important; }

/* risk bar */
.risk-wrap {
    border: 1px solid rgba(255,255,255,.09);
    background: rgba(255,255,255,.035);
    border-radius: 16px;
    padding: 20px 24px 16px 24px;
    margin: 6px 0 18px 0;
    animation: pageIn .5s cubic-bezier(.22,1,.36,1);
}
.risk-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 12px;
}
.risk-head .label { font-size: .95rem; opacity: .75; font-weight: 500; }
.risk-num {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -.04em;
    animation: popIn .5s cubic-bezier(.34,1.56,.64,1);
}
@keyframes popIn {
    from { opacity: 0; transform: scale(.82); }
    to   { opacity: 1; transform: scale(1); }
}
.risk-track {
    position: relative; height: 12px; border-radius: 99px;
    background: rgba(255,255,255,.08); overflow: hidden;
}
.risk-fill {
    height: 100%; border-radius: 99px;
    animation: grow 1s cubic-bezier(.22,1,.36,1);
    box-shadow: 0 0 18px currentColor;
}
@keyframes grow { from { width: 0 !important; } }
.risk-mark {
    position: absolute; top: -4px; width: 2px; height: 20px;
    background: rgba(255,255,255,.55);
}
.risk-scale {
    display: flex; justify-content: space-between;
    font-size: .74rem; opacity: .55; margin-top: 8px;
}

/* callout */
.callout {
    border-radius: 14px; padding: 16px 20px; margin: 4px 0 6px 0;
    border-left: 4px solid; line-height: 1.65; font-size: .97rem;
    animation: pageIn .5s cubic-bezier(.22,1,.36,1);
}

/* dataframe + map corners */
[data-testid="stDataFrame"], [data-testid="stDeckGlJsonChart"] {
    border-radius: 12px; overflow: hidden;
}

/* expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    font-weight: 600; border-radius: 10px;
}

#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def style(chart):
    """Consistent chart styling."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#9fb0c0", titleColor="#9fb0c0",
            domainColor="rgba(255,255,255,.15)", tickColor="rgba(255,255,255,.15)",
            grid=False, labelFont="Inter", titleFont="Inter",
            labelFontSize=12, titleFontSize=12, titleFontWeight=500,
        )
        .configure_legend(
            labelColor="#c6d2de", titleColor="#c6d2de",
            labelFont="Inter", titleFont="Inter", labelFontSize=12,
        )
    )


DATA = Path(__file__).parent / "data"
if not DATA.exists():
    DATA = Path(__file__).parent
TABLES = ["routes", "route_hours", "airlines", "airports", "climate",
          "airport_map", "hour_month", "weather_impact", "monthly_trend"]


# ---------- data + model loading ----------

@st.cache_resource
def get_db():
    con = duckdb.connect()
    for t in TABLES:
        path = (DATA / f"{t}.parquet").as_posix()
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{path}')")
    return con


def q(sql, params=None):
    cur = get_db().cursor()
    return cur.execute(sql, params or []).df()


@st.cache_resource
def load_json(name):
    return json.loads((DATA / name).read_text())


@st.cache_data(ttl=3600)
def forecast_weather(lat, lon, date_str, hour):
    """Open-Meteo forecast for a specific airport, date and hour. None if unavailable."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,snowfall,wind_gusts_10m,cloud_cover",
                "start_date": date_str, "end_date": date_str,
                "timezone": "auto",
            },
            timeout=6,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        return {
            "temp_c": float(h["temperature_2m"][hour]),
            "precip_mm": float(h["precipitation"][hour]),
            "snow_cm": float(h["snowfall"][hour]),
            "gust_kmh": float(h["wind_gusts_10m"][hour]),
            "cloud_pct": float(h["cloud_cover"][hour]),
        }
    except Exception:
        return None


MODEL = load_json("model.json")
STATS = load_json("stats.json")
NATIONAL = STATS["baseline_rate"]


def predict(df):
    """Scores rows with the exported logistic regression (same math as sklearn)."""
    z = np.full(len(df), MODEL["intercept"], dtype=float)
    for feat, table in MODEL["categorical"].items():
        z += df[feat].astype(str).map(table).fillna(0.0).to_numpy(dtype=float)
    for feat, p in MODEL["numeric"].items():
        x = df[feat].astype(float).to_numpy()
        if p.get("transform") == "log1p":
            x = np.log1p(np.clip(x, 0, None))
        z += p["coef"] * (x - p["mean"]) / p["scale"]
    return 1 / (1 + np.exp(-z))


def hour_label(h):
    h = int(h)
    suffix = "AM" if h < 12 else "PM"
    return f"{h % 12 or 12} {suffix}"


def risk_level(p):
    if p < NATIONAL * 0.8:
        return "Low"
    if p > NATIONAL * 1.25:
        return "High"
    return "Moderate"


RISK_COLOR = {"Low": GOOD, "Moderate": WARN, "High": ACCENT}


LIVE = "Live forecast (next 16 days)"

WEATHER_OPTIONS = [
    LIVE,
    "Typical for that month",
    "Clear skies",
    "Light rain",
    "Heavy rain or thunderstorms",
    "Snow",
    "Very windy",
]


def weather_values(choice, clim):
    """Turns a weather choice into the hourly weather features the model expects."""
    w = {
        "temp_c": clim["temp_c"],
        "precip_mm": clim["precip_mm"],
        "snow_cm": clim["snow_cm"],
        "gust_kmh": clim["gust_kmh"],
        "cloud_pct": clim["cloud_pct"],
    }
    if choice == "Clear skies":
        w.update(precip_mm=0.0, snow_cm=0.0, cloud_pct=10.0)
    elif choice == "Light rain":
        w.update(precip_mm=1.0, snow_cm=0.0, cloud_pct=90.0)
    elif choice == "Heavy rain or thunderstorms":
        w.update(precip_mm=6.0, snow_cm=0.0, cloud_pct=100.0, gust_kmh=max(clim["gust_kmh"], 45.0))
    elif choice == "Snow":
        w.update(precip_mm=1.5, snow_cm=1.5, cloud_pct=100.0, temp_c=min(clim["temp_c"], -2.0))
    elif choice == "Very windy":
        w.update(gust_kmh=65.0)
    return w


# ---------- lookups ----------

airports = q("SELECT * FROM airports").set_index("origin")


def airport_label(code):
    if code in airports.index:
        row = airports.loc[code]
        name = row.get("airport_name")
        city = row.get("city")
        if isinstance(name, str) and name:
            return f"{code} · {name}"
        if isinstance(city, str) and city:
            return f"{code} · {city}"
    return code


origins = q("""
    SELECT origin, SUM(flights) AS flights
    FROM routes GROUP BY origin ORDER BY origin
""")
origin_list = origins["origin"].tolist()
default_origin = "CLT" if "CLT" in origin_list else origins.sort_values("flights", ascending=False)["origin"].iloc[0]


# ---------- header ----------

st.markdown(f"""
<div class="hero">
  <h1><span class="plane">✈️</span> Flight Delay Risk Checker</h1>
  <p>
    Check the chance your flight leaves <b>15 or more minutes late</b>, or explore what drives delays.
    Estimates come from a model trained on <b>{STATS['flights']:,}</b> U.S. flights
    ({STATS['start']} to {STATS['end']}) combined with hourly weather at each departure airport.
  </p>
</div>
""", unsafe_allow_html=True)


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def weighted(df, by):
    out = df.groupby(by, as_index=False)[["flights", "delayed"]].sum()
    out["pct_delayed"] = out["delayed"] / out["flights"]
    return out


def render_insights():
    lb = q("SELECT * FROM airlines")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flights analyzed", f"{STATS['flights'] / 1e6:.1f}M")
    k2.metric("Left 15+ min late", f"{NATIONAL:.0%}")
    k3.metric("Airlines compared", f"{len(lb)}")
    k4.metric("Model ROC AUC", f"{STATS['auc']:.3f}")

    # Airline scorecard: actual vs adjusted
    st.subheader("Which airlines run late most often")
    order = lb.sort_values("adjusted_pct_delayed", ascending=False)["airline"].tolist()
    long = lb.melt(
        id_vars=["airline", "flights"],
        value_vars=["pct_delayed", "adjusted_pct_delayed"],
        var_name="measure", value_name="rate",
    )
    labels = {"pct_delayed": "Actual delay rate", "adjusted_pct_delayed": "Adjusted for airport, time, and weather"}
    long["measure"] = long["measure"].map(labels)
    chart = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("airline:N", sort=order, title=None),
        yOffset=alt.YOffset("measure:N", sort=list(labels.values())),
        x=alt.X("rate:Q", title="Share of flights 15+ minutes late", axis=alt.Axis(format="%", tickCount=6)),
        color=alt.Color("measure:N", sort=list(labels.values()),
                        scale=alt.Scale(range=[MUTED, ACCENT]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=400)),
        tooltip=[
            alt.Tooltip("airline:N", title="Airline"),
            alt.Tooltip("measure:N", title="Measure"),
            alt.Tooltip("rate:Q", title="Rate", format=".1%"),
            alt.Tooltip("flights:Q", title="Flights", format=","),
        ],
    ).properties(height=max(300, 38 * len(lb)))
    st.altair_chart(style(chart), width="stretch")
    st.caption(
        "The adjusted rate gives every airline the exact same set of flights (same airports, hours, months, "
        "and weather). Airlines that stay near the top after adjusting are not just unlucky with where and when they fly."
    )

    # Airport map + worst airports table
    st.subheader("Delay rate by departure airport")
    ap = q("SELECT * FROM airport_map WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    lo, hi = ap["pct_delayed"].min(), ap["pct_delayed"].max()
    t = ((ap["pct_delayed"] - lo) / (hi - lo if hi > lo else 1)).clip(0, 1)
    ap["color"] = [[255, int(210 * (1 - x)), int(80 * (1 - x)), 190] for x in t]
    ap["size"] = 9000 + 45000 * np.sqrt(ap["flights"] / ap["flights"].max())
    mcol, tcol = st.columns([2, 1])
    with mcol:
        st.map(ap, latitude="latitude", longitude="longitude", size="size", color="color")
        st.caption("Bigger circles mean more departures. Redder circles mean a higher delay rate.")
    with tcol:
        worst = ap.sort_values("pct_delayed", ascending=False).head(10)
        st.dataframe(
            worst[["origin", "flights", "pct_delayed"]].rename(
                columns={"origin": "Airport", "flights": "Flights", "pct_delayed": "Delay rate"}),
            hide_index=True, width="stretch",
            column_config={
                "Flights": st.column_config.NumberColumn(format="localized"),
                "Delay rate": st.column_config.NumberColumn(format="percent"),
            },
        )

    # Heatmap: hour x month
    st.subheader("When delays happen")
    hm = q("SELECT * FROM hour_month")
    hm["hour_label"] = hm["dep_hour"].map(hour_label)
    hm["month_name"] = hm["month"].map(lambda m: MONTHS[int(m) - 1])
    heat = alt.Chart(hm).mark_rect(cornerRadius=2).encode(
        x=alt.X("hour_label:N", sort=[hour_label(h) for h in range(5, 24)], title="Scheduled departure"),
        y=alt.Y("month_name:N", sort=MONTHS, title=None),
        color=alt.Color("pct_delayed:Q", scale=alt.Scale(scheme="orangered"),
                        legend=alt.Legend(title="Delay rate", format="%")),
        tooltip=[
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("hour_label:N", title="Departure"),
            alt.Tooltip("pct_delayed:Q", title="Delay rate", format=".1%"),
            alt.Tooltip("flights:Q", title="Flights", format=","),
        ],
    ).properties(height=360)
    st.altair_chart(style(heat), width="stretch")
    st.caption("Delays pile up as the day goes on and peak in summer, when thunderstorms and full schedules hit at the same time.")

    # Weather impact + monthly trend
    airline_names = sorted(lb["airline"].tolist())
    wcol, mcol2 = st.columns(2)
    with wcol:
        st.subheader("How weather changes the odds")
        pick = st.selectbox("Airline", ["All airlines"] + airline_names, key="wx_airline")
        wi = q("SELECT * FROM weather_impact")
        if pick != "All airlines":
            wi = wi[wi["airline"] == pick]
        wi = weighted(wi, ["condition", "condition_order"]).sort_values("condition_order")
        wchart = alt.Chart(wi).mark_bar(color=MUTED, cornerRadiusEnd=4).encode(
            x=alt.X("condition:N", sort=wi["condition"].tolist(), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pct_delayed:Q", title="Delay rate", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("condition:N", title="Weather at departure"),
                alt.Tooltip("pct_delayed:Q", title="Delay rate", format=".1%"),
                alt.Tooltip("flights:Q", title="Flights", format=","),
            ],
        ).properties(height=320)
        st.altair_chart(style(wchart), width="stretch")

    with mcol2:
        st.subheader("Delay rate over time")
        top = lb.sort_values("flights", ascending=False)["airline"].head(4).tolist()
        chosen = st.multiselect("Airlines", airline_names, default=top, key="trend_airlines")
        mt = q("SELECT * FROM monthly_trend")
        overall = weighted(mt, ["month_start"]).assign(airline="All airlines")
        sel = weighted(mt[mt["airline"].isin(chosen)], ["month_start", "airline"])
        both = pd.concat([overall, sel], ignore_index=True)
        tchart = alt.Chart(both).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("month_start:T", title=None),
            y=alt.Y("pct_delayed:Q", title="Delay rate", axis=alt.Axis(format="%")),
            color=alt.Color("airline:N", legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=250)),
            strokeDash=alt.condition(alt.datum.airline == "All airlines", alt.value([5, 4]), alt.value([0])),
            tooltip=[
                alt.Tooltip("airline:N", title="Airline"),
                alt.Tooltip("month_start:T", title="Month", format="%b %Y"),
                alt.Tooltip("pct_delayed:Q", title="Delay rate", format=".1%"),
            ],
        ).properties(height=320)
        st.altair_chart(style(tchart), width="stretch")


def render_checker():
    # ---------- inputs ----------

    c1, c2, c3 = st.columns(3)
    with c1:
        origin = st.selectbox("Departing from", origin_list, index=origin_list.index(default_origin), format_func=airport_label)

    dests = q("""
        SELECT dest, SUM(flights) AS flights
        FROM routes WHERE origin = ? GROUP BY dest ORDER BY flights DESC
    """, [origin])["dest"].tolist()

    with c2:
        dest = st.selectbox("Flying to", dests, format_func=airport_label)

    route_airlines = q("""
        SELECT airline, flights FROM routes
        WHERE origin = ? AND dest = ? ORDER BY flights DESC
    """, [origin, dest])["airline"].tolist()

    with c3:
        airline = st.selectbox("Airline", route_airlines)

    hours_flown = q("""
        SELECT dep_hour, flights FROM route_hours
        WHERE origin = ? AND dest = ? AND airline = ?
        ORDER BY dep_hour
    """, [origin, dest, airline])

    if len(hours_flown):
        default_hour = int(hours_flown.sort_values("flights", ascending=False)["dep_hour"].iloc[0])
    else:
        default_hour = 9

    c4, c5, c6 = st.columns(3)
    with c4:
        travel_date = st.date_input("Travel date", value=dt.date.today() + dt.timedelta(days=14))
    with c5:
        hour = st.selectbox("Scheduled departure", list(range(24)), index=default_hour, format_func=hour_label)
    with c6:
        weather_choice = st.selectbox("Expected weather at departure", WEATHER_OPTIONS)

    month = travel_date.month
    weekday = travel_date.strftime("%A")

    clim = q("SELECT * FROM climate WHERE origin = ? AND month = ?", [origin, month])
    if len(clim):
        clim = clim.iloc[0].to_dict()
    else:
        clim = {"temp_c": 15.0, "precip_mm": 0.0, "snow_cm": 0.0, "gust_kmh": 25.0, "cloud_pct": 50.0}
    forecast_note = None
    if weather_choice == LIVE:
        coords = q("SELECT latitude, longitude FROM airport_map WHERE origin = ?", [origin])
        live = None
        if len(coords) and pd.notna(coords.iloc[0]["latitude"]):
            live = forecast_weather(
                float(coords.iloc[0]["latitude"]), float(coords.iloc[0]["longitude"]),
                travel_date.isoformat(), hour,
            )
        if live:
            wx = live
            forecast_note = (
                f"Live Open-Meteo forecast for {origin} on {travel_date.strftime('%b %d')} "
                f"at {hour_label(hour)}."
            )
        else:
            wx = weather_values("Typical for that month", clim)
            forecast_note = (
                "No forecast available for that date, so this falls back to typical conditions for the month. "
                "Forecasts only reach about 16 days out."
            )
    else:
        wx = weather_values(weather_choice, clim)

    def build_rows(**overrides):
        base = {
            "airline": airline, "origin": origin, "dest": dest,
            "dep_hour": hour, "month": month, "day_of_week": weekday, **wx,
        }
        base.update(overrides)
        return base

    # ---------- main prediction ----------

    with st.spinner("Scoring this flight..."):
        p = float(predict(pd.DataFrame([build_rows()]))[0])
    level = risk_level(p)
    color = RISK_COLOR[level]

    route_stats = q("""
        SELECT * FROM routes WHERE origin = ? AND dest = ? AND airline = ?
    """, [origin, dest, airline]).iloc[0]

    st.divider()

    if forecast_note:
        st.caption(forecast_note)

    # animated risk bar (scale tops out at 50%)
    fill = min(p / 0.5, 1.0) * 100
    mark = min(NATIONAL / 0.5, 1.0) * 100
    st.markdown(f"""
    <div class="risk-wrap">
      <div class="risk-head">
        <span class="label">Chance of leaving 15+ minutes late &mdash; <b>{level.lower()} risk</b></span>
        <span class="risk-num" style="color:{color}">{p:.0%}</span>
      </div>
      <div class="risk-track">
        <div class="risk-fill" style="width:{fill:.1f}%; background:{color}; color:{color}"></div>
        <div class="risk-mark" style="left:{mark:.1f}%"></div>
      </div>
      <div class="risk-scale">
        <span>0%</span><span>national average {NATIONAL:.0%}</span><span>50%+</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Chance of a 15+ minute delay",
        f"{p:.0%}",
        delta=f"{(p - NATIONAL) * 100:+.0f} pts vs national average",
        delta_color="inverse",
    )
    m2.metric(
        "How often this route ran late",
        f"{route_stats['pct_delayed']:.0%}",
        help=f"Based on {int(route_stats['flights']):,} past {airline} flights from {origin} to {dest}.",
    )
    m3.metric(
        "Typical delay when it is late",
        f"{route_stats['avg_delay_when_late']:.0f} min",
        help=f"{route_stats['pct_severe']:.0%} of past flights on this route left more than an hour late.",
    )

    WEATHER_PHRASE = {
        LIVE: "the forecast conditions",
        "Typical for that month": "typical weather for the month",
        "Clear skies": "clear skies",
        "Light rain": "light rain",
        "Heavy rain or thunderstorms": "heavy rain or storms",
        "Snow": "snow",
        "Very windy": "strong winds",
    }
    st.markdown(f"""
    <div class="callout" style="border-color:{color}; background:{color}14">
      <b>{level} risk.</b> Flying {airline} from {origin} to {dest} at {hour_label(hour)}
      on a {weekday} in {travel_date.strftime('%B')}, with {WEATHER_PHRASE[weather_choice]},
      there is about a <b>{p:.0%}</b> chance of leaving 15+ minutes late. The national average is {NATIONAL:.0%}.
    </div>
    """, unsafe_allow_html=True)

    # ---------- chart 1: best time of day ----------

    left, right = st.columns(2)

    with left:
        st.subheader("Best time of day to fly this route")
        hrs = list(range(5, 24))
        hour_df = pd.DataFrame([build_rows(dep_hour=h) for h in hrs])
        hour_df["risk"] = predict(hour_df)
        hour_df["label"] = hour_df["dep_hour"].map(hour_label)
        hour_df["selected"] = np.where(hour_df["dep_hour"] == hour, "Your flight", "Other times")
        flown = set(hours_flown.loc[hours_flown["flights"] >= 10, "dep_hour"].astype(int))
        hour_df["scheduled"] = np.where(hour_df["dep_hour"].isin(flown), "Yes", "No")

        chart = alt.Chart(hour_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("label:N", sort=hour_df["label"].tolist(), title="Scheduled departure"),
            y=alt.Y("risk:Q", title="Chance of delay", axis=alt.Axis(format="%")),
            color=alt.Color("selected:N", scale=alt.Scale(domain=["Your flight", "Other times"], range=[ACCENT, MUTED]), legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Departure"),
                alt.Tooltip("risk:Q", title="Chance of delay", format=".0%"),
                alt.Tooltip("scheduled:N", title=f"{airline} usually flies then"),
            ],
        ).properties(height=320)
        st.altair_chart(style(chart), width="stretch")

        pool = hour_df[hour_df["dep_hour"].isin(flown)] if flown else hour_df
        best = pool.sort_values("risk").iloc[0]
        st.caption(
            f"Lowest risk departure time {'that ' + airline + ' actually schedules ' if flown else ''}"
            f"on this route: **{best['label']}** ({best['risk']:.0%}). Delays tend to build as the day goes on."
        )

    # ---------- chart 2: compare airlines on this route ----------

    with right:
        st.subheader("Airlines on this route")
        comp = q("""
            SELECT airline, flights, pct_delayed FROM routes
            WHERE origin = ? AND dest = ?
        """, [origin, dest])
        comp_rows = pd.DataFrame([build_rows(airline=a) for a in comp["airline"]])
        comp["risk"] = predict(comp_rows)
        comp["selected"] = np.where(comp["airline"] == airline, "Your airline", "Other airlines")
        comp = comp.sort_values("risk")

        chart2 = alt.Chart(comp).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("airline:N", sort=comp["airline"].tolist(), title=None),
            x=alt.X("risk:Q", title="Chance of delay", axis=alt.Axis(format="%")),
            color=alt.Color("selected:N", scale=alt.Scale(domain=["Your airline", "Other airlines"], range=[ACCENT, MUTED]), legend=None),
            tooltip=[
                alt.Tooltip("airline:N", title="Airline"),
                alt.Tooltip("risk:Q", title="Estimated chance of delay", format=".0%"),
                alt.Tooltip("pct_delayed:Q", title="Historical delay rate", format=".0%"),
                alt.Tooltip("flights:Q", title="Past flights", format=","),
            ],
        ).properties(height=max(160, 45 * len(comp)))
        st.altair_chart(style(chart2), width="stretch")

        if len(comp) > 1:
            st.caption(f"Same time, date, and weather for every airline. Lowest risk here: **{comp.iloc[0]['airline']}**.")
        else:
            st.caption(f"{airline} is the only airline in the data flying this route.")

    # ---------- extra sections ----------
    with st.expander("How this works"):
        st.markdown(f"""
    The estimate comes from a logistic regression model trained on **{STATS['train_rows']:,}** flights sampled from
    **{STATS['flights']:,}** U.S. domestic departures reported to the Bureau of Transportation Statistics.
    Each flight was matched to the hourly weather at its departure airport using the Open-Meteo historical archive.

    **What the model looks at:** airline, departure airport, destination, scheduled hour, month, day of week,
    temperature, precipitation, snowfall, wind gusts, and cloud cover.

    **How good is it:** ROC AUC of **{STATS['auc']:.3f}** on held-out flights. That means it ranks a delayed flight
    as riskier than an on-time flight about {STATS['auc']:.0%} of the time. Delays have a lot of randomness
    (a late inbound plane, a crew issue), so treat this as a risk estimate, not a guarantee.

    **Weather used for this estimate:** {wx['temp_c']:.0f}°C, {wx['precip_mm']:.1f} mm/hr precipitation,
    {wx['snow_cm']:.1f} cm/hr snow, {wx['gust_kmh']:.0f} km/h gusts, {wx['cloud_pct']:.0f}% cloud cover.
    "Typical for that month" uses the average conditions at {origin} in {travel_date.strftime('%B')}.
    "Live forecast" pulls the Open-Meteo forecast for your airport, date and hour. Worth noting that the model was
    trained on observed historical weather, and a forecast is not an observation, so a prediction based on a
    two-week-out forecast carries the forecast's own error on top of the model's.

    **Coverage:** departures from the {len(origin_list)} airports with weather data, on routes with at least
    {STATS['min_route_flights']} flights in the data. Cancelled flights are not included.
    """)


tab1, tab2 = st.tabs(["Check a flight", "Delay insights"])
with tab1:
    render_checker()
with tab2:
    render_insights()

st.caption("Data: U.S. Bureau of Transportation Statistics on-time performance, Open-Meteo historical weather. Built by Rishabh Kota.")
