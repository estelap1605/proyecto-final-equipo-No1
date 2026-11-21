
import pandas as pd
import yaml
import requests

from pathlib import Path


API_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_weather():

    config = load_config()

    raw_path = Path(config["data"]["raw_path"])
    initial_path = raw_path / "initial_dataset.csv"

    if not initial_path.exists():
        raise FileNotFoundError(
            "Primero ejecuta download_initial.py"
        )

    initial = pd.read_csv(initial_path)

    date_col = config["data"]["initial"]["date_column"]

    initial[date_col] = pd.to_datetime(initial[date_col])

    start_date = initial[date_col].min().strftime("%Y-%m-%d")
    end_date = initial[date_col].max().strftime("%Y-%m-%d")

    weather_config = config["data"]["weather"]

    # Variables configuradas
    daily_variables = weather_config["daily_variables"]
    hourly_variables = weather_config["hourly_variables"]

    # Parámetros de Open-Meteo
    params = {
        "latitude": weather_config["latitude"],
        "longitude": weather_config["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_variables),
        "hourly": ",".join(hourly_variables),
        "timezone": weather_config["timezone"],
        "temperature_unit": weather_config["temperature_unit"],
        "wind_speed_unit": weather_config["wind_speed_unit"],
        "precipitation_unit": weather_config["precipitation_unit"]
    }

    print("\nDESCARGANDO CLIMA HISTÓRICO")
    print("Desde:", start_date)
    print("Hasta:", end_date)

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    if "daily" not in result:
        raise ValueError("La API no devolvió datos diarios.")

    # 1. Procesar variables diarias
    daily = pd.DataFrame(result["daily"])

    daily.rename(
        columns={"time": "date"},
        inplace=True
    )

    daily["date"] = pd.to_datetime(daily["date"])

    # 2. Procesar humedad horaria
    humidity = pd.DataFrame(result["hourly"])

    humidity["date"] = pd.to_datetime(
        humidity["time"]
    ).dt.normalize()

    humidity_daily = (
        humidity.groupby("date")[
            "relative_humidity_2m"
        ]
        .agg(
            relative_humidity_2m_mean="mean",
            humidity_observations="count"
        )
        .reset_index()
    )

    # 3. Integrar humedad con clima diario
    weather = daily.merge(
        humidity_daily,
        on="date",
        how="left",
        validate="one_to_one"
    )

    # 4. Guardar metadatos
    metadata = {
        "latitude": weather_config["latitude"],
        "longitude": weather_config["longitude"],
        "timezone": weather_config["timezone"],
        "start_date": start_date,
        "end_date": end_date,
        "api": API_URL
    }

    weather_path = raw_path / "istanbul_weather.csv"
    metadata_path = raw_path / "weather_metadata.yaml"

    weather.to_csv(weather_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            metadata,
            file,
            allow_unicode=True,
            sort_keys=False
        )

    # 5. Reporte
    print("\nCLIMA DESCARGADO")
    print("----------------")
    print("Registros:", len(weather))
    print("Fecha inicial:", weather["date"].min())
    print("Fecha final:", weather["date"].max())
    print("Variables:", list(weather.columns))
    print("Archivo:", weather_path)


if __name__ == "__main__":
    download_weather()
