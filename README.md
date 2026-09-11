# Flight Delay Risk Checker

A web app that estimates the chance a U.S. domestic flight leaves 15 or more minutes late, built on
10,269,974 flights from Mar 2024 to Jul 2025 joined with hourly weather at each departure airport.

**Live app:** ADD_STREAMLIT_LINK_HERE

## What it does

The app has two tabs.

**Check a flight:** pick an origin, destination, airline, travel date, departure time, and expected weather to get:

- The estimated chance of a 15+ minute delay, compared to the national average
- How often that exact route and airline actually ran late, and how long delays usually last
- The lowest risk time of day to fly that route
- How every airline on the route compares under the same conditions

**Delay insights:** an interactive dashboard with an airline scorecard (actual versus adjusted delay rates), a map of
delay rates by airport, an hour by month heatmap, the effect of weather, and monthly trends by airline.

It covers 8,684 route and airline combinations departing from 81 major airports.

## Key findings

- **Airline matters even after controlling for conditions.** Hawaiian Airlines had the highest delay rate at
  12.4%, and still ranked worst (35.8%) after adjusting for airport,
  time of day, month, and weather. Envoy Air ranked best at 18.4%.
- **Weather drives when delays happen.** Flights left late 19.7% of the time with no precipitation versus
  37.4% in heavy rain.
- **Delays snowball through the day.** 5 AM departures were late 7.5% of the time compared with
  32.0% at 8 PM.
- **Model performance:** logistic regression with ROC AUC of 0.714 on held-out flights.

## How it was built

1. **Data collection:** BTS on-time performance data, an airport reference table, and hourly weather from the
   Open-Meteo historical archive API.
2. **Database:** loaded into a DuckDB SQL database with flights, weather, and airports tables and a joined
   flights_weather view.
3. **SQL analysis:** airline scorecards, delay patterns by hour, month, route, and weather, and window function
   rankings of the most reliable airline at each airport.
4. **Modeling:** logistic regression on 2,000,000 sampled flights using airline, origin, destination,
   hour, month, day of week, and five weather features. An adjusted ranking gives every airline the same set of
   flights to separate airline performance from where and when they fly.
5. **Product:** the model is exported as plain coefficients and served in a Streamlit app that queries summary
   tables with DuckDB, with an insights dashboard built into the same app.

## Tech stack

Python, pandas, SQL (DuckDB), scikit-learn, Streamlit, Altair, Google Colab

## Repo structure

```
app.py              Streamlit app
requirements.txt    Python packages for deployment
data/               Summary tables (parquet), model coefficients, dataset stats
```

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

- U.S. Bureau of Transportation Statistics, Airline On-Time Performance
- Open-Meteo Historical Weather API
- airportsdata (airport reference)
