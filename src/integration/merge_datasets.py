
import pandas as pd
import yaml

from pathlib import Path


# ==========================================================
# CONFIGURACION
# ==========================================================

def load_config():

    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuracion: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("La configuracion YAML no es valida.")

    return config


# ==========================================================
# VALIDACIONES GENERALES
# ==========================================================

def validate_dates(df, dataset_name):

    if "date" not in df.columns:
        raise ValueError(
            f"{dataset_name}: no existe la columna date."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.normalize()

    if df["date"].isna().any():
        raise ValueError(
            f"{dataset_name}: existen fechas invalidas."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"{dataset_name}: existen fechas duplicadas."
        )

    return df.sort_values("date").reset_index(drop=True)


def validate_daily_sales(sales, target_col):

    if target_col not in sales.columns:
        raise ValueError(
            f"No existe la variable objetivo: {target_col}"
        )

    sales[target_col] = pd.to_numeric(
        sales[target_col],
        errors="coerce"
    )

    if sales[target_col].isna().any():
        raise ValueError(
            "Existen valores invalidos en las ventas."
        )

    expected_dates = pd.date_range(
        start=sales["date"].min(),
        end=sales["date"].max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(
        sales["date"]
    )

    if len(missing_dates) > 0:
        raise ValueError(
            "Existen dias sin registros de ventas: "
            f"{len(missing_dates)}. "
            f"Ejemplos: {list(missing_dates[:5])}"
        )

    return sales


# ==========================================================
# AGREGACION DE VARIABLES CLIMATICAS
# ==========================================================

def get_weather_aggregation(column):

    name = column.lower()

    if any(word in name for word in [
        "precipitation",
        "rain",
        "snow"
    ]):
        return "sum"

    if any(word in name for word in [
        "wind",
        "gust"
    ]):
        return "max"

    if "observation" in name:
        return "sum"

    # Temperatura y humedad:
    # promedio semanal
    return "mean"


# ==========================================================
# INTEGRACION DE DATASETS
# ==========================================================

def merge_datasets():

    config = load_config()

    data_config = config["data"]
    initial_config = data_config["initial"]
    modeling_config = config.get("modeling", {})

    if modeling_config.get("granularity", "weekly") != "weekly":
        raise ValueError(
            "Este modulo esta configurado para granularidad weekly."
        )

    raw_path = Path(data_config["raw_path"])

    output_path = Path(
        modeling_config.get(
            "output_file",
            "data/processed/modeling_dataset.csv"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    target_col = initial_config["target_column"]

    # ------------------------------------------------------
    # 1. CARGAR DATASETS
    # ------------------------------------------------------

    sales_path = raw_path / "initial_dataset.csv"
    weather_path = raw_path / "istanbul_weather.csv"
    holidays_path = raw_path / "turkey_holidays.csv"

    required_files = [
        sales_path,
        weather_path,
        holidays_path
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"No se encuentra el archivo: {path}"
            )

    sales = pd.read_csv(sales_path)
    weather = pd.read_csv(weather_path)
    holidays = pd.read_csv(holidays_path)

    # Normalizar nombre de columna de fecha
    date_col = initial_config["date_column"]

    if date_col != "date":
        sales.rename(
            columns={date_col: "date"},
            inplace=True
        )

    # ------------------------------------------------------
    # 2. VALIDAR FECHAS Y VENTAS
    # ------------------------------------------------------

    sales = validate_dates(
        sales,
        "Ventas"
    )

    weather = validate_dates(
        weather,
        "Clima"
    )

    holidays = validate_dates(
        holidays,
        "Festivos"
    )

    sales = validate_daily_sales(
        sales,
        target_col
    )

    start_date = sales["date"].min()
    end_date = sales["date"].max()

    expected_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    # ------------------------------------------------------
    # 3. VALIDAR COBERTURA DE CLIMA Y FESTIVOS
    # ------------------------------------------------------

    for name, dataset in [
        ("Clima", weather),
        ("Festivos", holidays)
    ]:

        available_dates = set(dataset["date"])

        missing_dates = set(expected_dates) - available_dates

        if missing_dates:
            raise ValueError(
                f"{name}: faltan {len(missing_dates)} dias "
                f"en el periodo de ventas."
            )

    # Seleccionar solamente el periodo de ventas
    weather = weather[
        weather["date"].between(start_date, end_date)
    ].copy()

    holidays = holidays[
        holidays["date"].between(start_date, end_date)
    ].copy()

    # Validar variables obligatorias
    required_holiday_columns = [
        "is_holiday",
        "is_weekend"
    ]

    missing_holiday_columns = [
        col for col in required_holiday_columns
        if col not in holidays.columns
    ]

    if missing_holiday_columns:
        raise ValueError(
            "Faltan columnas de calendario: "
            f"{missing_holiday_columns}"
        )

    if "daily_variables" in data_config["weather"]:
        required_weather = (
            data_config["weather"]["daily_variables"]
            + ["relative_humidity_2m_mean"]
        )

        missing_weather = [
            col for col in required_weather
            if col not in weather.columns
        ]

        if missing_weather:
            raise ValueError(
                f"Faltan variables climaticas: {missing_weather}"
            )

    # ------------------------------------------------------
    # 4. INTEGRAR VENTAS + CLIMA + FESTIVOS
    # ------------------------------------------------------

    daily = sales.merge(
        weather,
        on="date",
        how="left",
        validate="one_to_one",
        suffixes=("", "_weather")
    )

    daily = daily.merge(
        holidays,
        on="date",
        how="left",
        validate="one_to_one",
        suffixes=("", "_holiday")
    )

    if len(daily) != len(sales):
        raise ValueError(
            "El numero de registros cambio durante la integracion."
        )

    if daily[
        ["is_holiday", "is_weekend"]
    ].isna().any().any():

        raise ValueError(
            "Existen valores faltantes en el calendario."
        )

    # No imputar clima aqui.
    # Los valores faltantes se reportan para su tratamiento
    # en la etapa de limpieza.

    # ------------------------------------------------------
    # 5. PREPARAR VARIABLES PARA AGREGACION SEMANAL
    # ------------------------------------------------------

    daily["week_start"] = (
        daily["date"]
        - pd.to_timedelta(
            daily["date"].dt.weekday,
            unit="D"
        )
    )

    # Identificar variables climaticas numericas
    excluded_columns = {
        "date",
        "week_start",
        "holiday_name",
        "is_holiday",
        "is_weekend",
        "day_of_week",
        "month",
        "year"
    }

    weather_columns = [
        col for col in weather.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            daily[col]
        )
    ]

    # ------------------------------------------------------
    # 6. DEFINIR REGLAS DE AGREGACION SEMANAL
    # ------------------------------------------------------

    aggregations = {
        target_col: "sum",
        "date": "max",
        "is_holiday": "sum",
        "is_weekend": "sum"
    }

    for column in weather_columns:

        aggregations[column] = (
            get_weather_aggregation(column)
        )

    if "holiday_name" in daily.columns:

        aggregations["holiday_name"] = (
            lambda values: " | ".join(
                sorted({
                    str(value).strip()
                    for value in values.dropna()
                    if str(value).strip()
                })
            )
        )

    # ------------------------------------------------------
    # 7. GENERAR DATASET SEMANAL
    # ------------------------------------------------------

    weekly = (
        daily.groupby("week_start")
        .agg(aggregations)
        .reset_index()
    )

    weekly.rename(
        columns={
            "date": "week_end",
            "is_holiday": "holiday_days",
            "is_weekend": "weekend_days"
        },
        inplace=True
    )

    # Cantidad real de dias incluidos en cada semana
    days_per_week = (
        daily.groupby("week_start")
        .size()
        .rename("days_in_period")
        .reset_index()
    )

    weekly = weekly.merge(
        days_per_week,
        on="week_start",
        how="left",
        validate="one_to_one"
    )

    weekly["has_holiday"] = (
        weekly["holiday_days"] > 0
    ).astype(int)

    weekly["is_partial_week"] = (
        weekly["days_in_period"] < 7
    ).astype(int)

    # Ordenar cronologicamente
    weekly = weekly.sort_values(
        "week_start"
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # 8. VALIDAR DATASET FINAL
    # ------------------------------------------------------

    if weekly.empty:
        raise ValueError(
            "El dataset semanal esta vacio."
        )

    if weekly[target_col].isna().any():
        raise ValueError(
            "Existen valores faltantes en las ventas semanales."
        )

    if weekly["week_start"].duplicated().any():
        raise ValueError(
            "Existen semanas duplicadas."
        )

    # ------------------------------------------------------
    # 9. GUARDAR DATASET
    # ------------------------------------------------------

    weekly.to_csv(
        output_path,
        index=False
    )

    # ------------------------------------------------------
    # 10. REPORTE
    # ------------------------------------------------------

    forecast_days = modeling_config.get(
        "forecast_horizon_days",
        30
    )

    print("\n========================================")
    print(" DATASET SEMANAL DE MODELADO GENERADO")
    print("========================================")

    print("Registros diarios:", len(daily))
    print("Registros semanales:", len(weekly))

    print("Fecha inicial:", start_date.date())
    print("Fecha final:", end_date.date())

    print("Horizonte de pronostico:", forecast_days, "dias")
    print("Granularidad: semanal")

    print(
        "Periodos semanales aproximados:",
        (forecast_days + 6) // 7
    )

    print("Ventas totales:", weekly[target_col].sum())

    print(
        "Semanas parciales:",
        weekly["is_partial_week"].sum()
    )

    print(
        "Valores faltantes por columna:"
    )

    print(
        weekly.isna().sum()[
            weekly.isna().sum() > 0
        ]
    )

    print("\nColumnas generadas:")
    print(weekly.columns.tolist())

    print("\nPrimeros registros:")
    print(weekly.head().to_string(index=False))

    print("\nArchivo generado:", output_path)

    print("\nINTEGRACION COMPLETADA")
    print("========================================")


if __name__ == "__main__":
    merge_datasets()
