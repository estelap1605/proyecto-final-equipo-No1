
import pandas as pd
import yaml
import holidays

from pathlib import Path


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_holidays():

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

    start_date = initial[date_col].min()
    end_date = initial[date_col].max()

    start_year = start_date.year
    end_year = end_date.year

    holiday_config = config["data"]["holidays"]

    # 1. Calendario oficial del país
    turkey_calendar = holidays.country_holidays(
        holiday_config["country"],
        years=range(start_year, end_year + 1),
        language=holiday_config["language"]
    )

    # 2. Generar calendario diario
    dates = pd.date_range(
        start=start_date.normalize(),
        end=end_date.normalize(),
        freq="D"
    )

    records = []

    for date in dates:

        holiday_name = turkey_calendar.get(date.date())

        records.append({
            "date": date,
            "is_holiday": int(holiday_name is not None),
            "holiday_name": holiday_name,
            "is_weekend": int(date.weekday() >= 5),
            "day_of_week": date.dayofweek,
            "month": date.month,
            "year": date.year
        })

    df = pd.DataFrame(records)

    # 3. Guardar
    output_path = raw_path / "turkey_holidays.csv"

    df.to_csv(output_path, index=False)

    # 4. Reporte
    print("\nCALENDARIO DE TURQUÍA")
    print("---------------------")
    print("Año inicial:", start_year)
    print("Año final:", end_year)
    print("Días generados:", len(df))
    print("Festivos encontrados:", df["is_holiday"].sum())
    print("Fines de semana:", df["is_weekend"].sum())

    print("\nFestivos:")
    print(
        df[df["is_holiday"] == 1][
            ["date", "holiday_name"]
        ].to_string(index=False)
    )

    print("\nArchivo:", output_path)


if __name__ == "__main__":
    download_holidays()
