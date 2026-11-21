
import pandas as pd
import yaml

from pathlib import Path


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_sources():

    config = load_config()

    raw_path = Path(config["data"]["raw_path"])

    initial_file = raw_path / "initial_dataset.csv"
    weather_file = raw_path / "istanbul_weather.csv"
    holidays_file = raw_path / "turkey_holidays.csv"

    files = {
        "initial": initial_file,
        "weather": weather_file,
        "holidays": holidays_file
    }

    # 1. Verificar existencia
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el archivo {name}: {path}"
            )

    # 2. Cargar datasets
    initial = pd.read_csv(initial_file)
    weather = pd.read_csv(weather_file)
    holidays = pd.read_csv(holidays_file)

    date_col = config["data"]["initial"]["date_column"]
    target_col = config["data"]["initial"]["target_column"]

    # 3. Verificar columnas
    required_columns = {
        "initial": [date_col, target_col],
        "weather": [
            "date",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean"
        ],
        "holidays": [
            "date",
            "is_holiday",
            "is_weekend"
        ]
    }

    datasets = {
        "initial": initial,
        "weather": weather,
        "holidays": holidays
    }

    errors = []

    for name, df in datasets.items():

        missing = [
            col for col in required_columns[name]
            if col not in df.columns
        ]

        if missing:
            errors.append(
                f"{name}: faltan columnas {missing}"
            )

    if errors:
        raise ValueError("\n".join(errors))

    # 4. Convertir fechas
    initial[date_col] = pd.to_datetime(
        initial[date_col],
        errors="coerce"
    )

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce"
    )

    holidays["date"] = pd.to_datetime(
        holidays["date"],
        errors="coerce"
    )

    for name, df, col in [
        ("initial", initial, date_col),
        ("weather", weather, "date"),
        ("holidays", holidays, "date")
    ]:
        if df[col].isna().any():
            errors.append(
                f"{name}: existen fechas inválidas"
            )

    if errors:
        raise ValueError("\n".join(errors))

    # 5. Normalizar fechas para comparar días
    initial_days = initial[date_col].dt.normalize()

    start = initial_days.min()
    end = initial_days.max()

    # 6. Validar duplicados
    # El dataset inicial puede tener varias observaciones
    # por día si su granularidad es horaria.
    if initial[date_col].duplicated().any():
        errors.append(
            "initial: existen timestamps duplicados"
        )

    if weather["date"].dt.normalize().duplicated().any():
        errors.append(
            "weather: existen días duplicados"
        )

    if holidays["date"].dt.normalize().duplicated().any():
        errors.append(
            "holidays: existen días duplicados"
        )

    # 7. Validar orden cronológico
    if not initial[date_col].is_monotonic_increasing:
        errors.append("initial: fechas desordenadas")

    if not weather["date"].is_monotonic_increasing:
        errors.append("weather: fechas desordenadas")

    if not holidays["date"].is_monotonic_increasing:
        errors.append("holidays: fechas desordenadas")

    # 8. Calcular cobertura
    expected_days = pd.date_range(
        start=start,
        end=end,
        freq="D"
    )

    weather_dates = set(
        weather["date"].dt.normalize()
    )

    holiday_dates = set(
        holidays["date"].dt.normalize()
    )

    weather_coverage = (
        len(set(expected_days) & weather_dates)
        / len(expected_days)
    )

    holiday_coverage = (
        len(set(expected_days) & holiday_dates)
        / len(expected_days)
    )

    # 9. Validar umbrales
    min_weather = config["validation"][
        "min_weather_coverage"
    ]

    min_holidays = config["validation"][
        "min_holiday_coverage"
    ]

    if weather_coverage < min_weather:
        errors.append(
            f"Cobertura climática insuficiente: "
            f"{weather_coverage:.2%}"
        )

    if holiday_coverage < min_holidays:
        errors.append(
            f"Cobertura calendario insuficiente: "
            f"{holiday_coverage:.2%}"
        )

    # 10. Reporte de nulos
    print("\nREPORTE DE CALIDAD")
    print("==================")

    for name, df in datasets.items():

        print(f"\nDataset: {name}")
        print("Registros:", len(df))
        print("Columnas:", len(df.columns))
        print("Valores faltantes:", int(df.isna().sum().sum()))

    print("\nRANGO HISTÓRICO")
    print("Inicio:", start)
    print("Final:", end)

    print("\nCOBERTURA")
    print(f"Clima: {weather_coverage:.2%}")
    print(f"Festivos: {holiday_coverage:.2%}")

    print("\nFECHAS SIN CLIMA:")
    print(
        sorted(set(expected_days) - weather_dates)[:20]
    )

    print("\nFECHAS SIN CALENDARIO:")
    print(
        sorted(set(expected_days) - holiday_dates)[:20]
    )

    if errors:
        print("\nVALIDACIÓN FALLIDA")
        for error in errors:
            print("-", error)

        raise ValueError(
            "Corregir los problemas antes de continuar."
        )

    print("\nVALIDACIÓN EXITOSA")
    print("Los tres datasets están disponibles.")


if __name__ == "__main__":
    validate_sources()
