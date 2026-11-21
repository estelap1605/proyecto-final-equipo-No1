
from pathlib import Path

import pandas as pd
import yaml


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
# VALIDACIONES
# ==========================================================

def normalize_dates(df, dataset_name):
    if "date" not in df.columns:
        raise ValueError(
            f"{dataset_name}: no existe la columna date."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()

    if df["date"].isna().any():
        raise ValueError(
            f"{dataset_name}: existen fechas invalidas."
        )

    return df.sort_values("date").reset_index(drop=True)


def validate_unique_dates(df, dataset_name):
    if df["date"].duplicated().any():
        raise ValueError(
            f"{dataset_name}: existen fechas duplicadas."
        )


def get_weather_aggregation(column):
    name = column.lower()

    if any(word in name for word in [
        "precipitation", "rain", "snow", "observation"
    ]):
        return "sum"

    if "wind" in name or "gust" in name:
        return "max"

    return "mean"


def join_names(values):
    names = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    }
    return " | ".join(sorted(names))


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
            "Este modulo requiere granularidad weekly."
        )

    raw_path = Path(data_config["raw_path"])

    output_path = Path(
        modeling_config.get(
            "output_file",
            "data/processed/modeling_dataset.csv"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_col = initial_config.get("target_column", "quantity")

    # ------------------------------------------------------
    # 1. CARGAR ARCHIVOS
    # ------------------------------------------------------

    sales_path = Path(
        initial_config.get(
            "file_path",
            "data/raw/initial_dataset.csv"
        )
    )
    weather_path = raw_path / "istanbul_weather.csv"
    holidays_path = raw_path / "turkey_holidays.csv"

    for path in [sales_path, weather_path, holidays_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encuentra el archivo: {path}"
            )

    sales = pd.read_csv(sales_path)
    weather = pd.read_csv(weather_path)
    holidays = pd.read_csv(holidays_path)

    # ------------------------------------------------------
    # 2. NORMALIZAR Y VALIDAR FECHAS
    # ------------------------------------------------------

    date_col = initial_config.get("date_column", "date")

    if date_col not in sales.columns:
        raise ValueError(
            f"No existe la columna de fecha {date_col} en ventas."
        )

    if date_col != "date":
        sales = sales.rename(columns={date_col: "date"})

    sales = normalize_dates(sales, "Ventas")
    weather = normalize_dates(weather, "Clima")
    holidays = normalize_dates(holidays, "Festivos")

    # Ventas puede tener muchas filas por fecha.
    # Clima y calendario deben tener una sola fila por fecha.
    validate_unique_dates(weather, "Clima")
    validate_unique_dates(holidays, "Festivos")

    # ------------------------------------------------------
    # 3. VALIDAR COLUMNAS DE VENTAS
    # ------------------------------------------------------

    dimensions = [
        "shopping_mall",
        "category",
        "gender"
    ]

    required_sales = ["date", target_col] + dimensions

    missing_columns = [
        col for col in required_sales
        if col not in sales.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en ventas: {missing_columns}"
        )

    sales[target_col] = pd.to_numeric(
        sales[target_col], errors="coerce"
    )

    if sales[target_col].isna().any():
        raise ValueError(
            f"Hay valores invalidos en {target_col}."
        )

    if (sales[target_col] < 0).any():
        raise ValueError(
            f"Hay valores negativos en {target_col}."
        )

    for col in dimensions:
        sales[col] = sales[col].astype("string").str.strip()

        if sales[col].isna().any() or (sales[col] == "").any():
            raise ValueError(
                f"Hay valores vacios en la dimension {col}."
            )

    # Validar ventas y transacciones
    if "sales" in sales.columns:
        sales["sales"] = pd.to_numeric(
            sales["sales"], errors="coerce"
        )
        if sales["sales"].isna().any():
            raise ValueError("Hay valores invalidos en sales.")

    if "transactions" not in sales.columns:
        sales["transactions"] = 0

    sales["transactions"] = pd.to_numeric(
        sales["transactions"], errors="coerce"
    ).fillna(0)

    # Consolidar por si existen claves repetidas
    aggregations = {
        target_col: "sum",
        "transactions": "sum"
    }

    if "sales" in sales.columns:
        aggregations["sales"] = "sum"

    sales = (
        sales.groupby(
            ["date"] + dimensions,
            as_index=False,
            dropna=False
        )
        .agg(aggregations)
    )

    # ------------------------------------------------------
    # 4. VALIDAR COBERTURA DE CLIMA Y CALENDARIO
    # ------------------------------------------------------

    start_date = sales["date"].min()
    end_date = sales["date"].max()

    expected_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    for name, dataset in [
        ("Clima", weather),
        ("Festivos", holidays)
    ]:
        available_dates = set(dataset["date"])
        missing_dates = set(expected_dates) - available_dates

        if missing_dates:
            raise ValueError(
                f"{name}: faltan {len(missing_dates)} dias. "
                f"Ejemplos: {list(sorted(missing_dates)[:5])}"
            )

    weather = weather[
        weather["date"].between(start_date, end_date)
    ].copy()

    holidays = holidays[
        holidays["date"].between(start_date, end_date)
    ].copy()

    required_calendar = ["is_holiday", "is_weekend"]
    missing_calendar = [
        col for col in required_calendar
        if col not in holidays.columns
    ]

    if missing_calendar:
        raise ValueError(
            f"Faltan columnas de calendario: {missing_calendar}"
        )

    # ------------------------------------------------------
    # 5. COMPLETAR PANEL DIARIO DE DEMANDA
    # ------------------------------------------------------

    # Se conservan las combinaciones que aparecen en ventas.
    combinations = sales[dimensions].drop_duplicates()

    dates = pd.DataFrame({"date": expected_dates})

    panel = combinations.merge(dates, how="cross")

    daily = panel.merge(
        sales,
        on=["date"] + dimensions,
        how="left",
        validate="one_to_one"
    )

    # Un dia sin transacciones equivale a demanda cero.
    daily[target_col] = daily[target_col].fillna(0)
    daily["transactions"] = daily["transactions"].fillna(0)

    if "sales" in daily.columns:
        daily["sales"] = daily["sales"].fillna(0)

    # ------------------------------------------------------
    # 6. INTEGRAR CLIMA Y CALENDARIO A NIVEL DIARIO
    # ------------------------------------------------------

    daily = daily.merge(
        weather,
        on="date",
        how="left",
        validate="many_to_one"
    )

    daily = daily.merge(
        holidays,
        on="date",
        how="left",
        validate="many_to_one",
        suffixes=("", "_holiday")
    )

    if daily[required_calendar].isna().any().any():
        raise ValueError(
            "Hay valores faltantes en las variables del calendario."
        )

    # No imputar clima en esta etapa.
    # Los faltantes se reportan para su tratamiento posterior.

    # ------------------------------------------------------
    # 7. DEFINIR SEMANAS (LUNES A DOMINGO)
    # ------------------------------------------------------

    week_start_day = modeling_config.get("week_start", "MON")

    if week_start_day != "MON":
        raise ValueError(
            "Actualmente el modulo requiere semanas iniciando el lunes."
        )

    daily["week_start"] = (
        daily["date"]
        - pd.to_timedelta(daily["date"].dt.day_of_week, unit="D")
    )

    # ------------------------------------------------------
    # 8. AGREGAR VENTAS POR DIMENSIONES Y SEMANA
    # ------------------------------------------------------

    sales_aggregations = {
        target_col: "sum",
        "transactions": "sum"
    }

    if "sales" in daily.columns:
        sales_aggregations["sales"] = "sum"

    weekly_sales = (
        daily.groupby(
            ["week_start"] + dimensions,
            as_index=False
        )
        .agg(sales_aggregations)
    )

    
    # ------------------------------------------------------
    # 9. AGREGAR CLIMA Y CALENDARIO UNA SOLA VEZ POR SEMANA
    # ------------------------------------------------------

    features_daily = weather.merge(
        holidays,
        on="date",
        how="left",
        validate="one_to_one",
        suffixes=("", "_holiday")
    )

    features_daily["week_start"] = (
        features_daily["date"]
        - pd.to_timedelta(
            features_daily["date"].dt.day_of_week,
            unit="D"
        )
    )

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
        and pd.api.types.is_numeric_dtype(weather[col])
    ]

    weekly_aggregations = {
        "date": "max",
        "is_holiday": "sum",
        "is_weekend": "sum"
    }

    for col in weather_columns:
        weekly_aggregations[col] = get_weather_aggregation(col)

    if "holiday_name" in features_daily.columns:
        weekly_aggregations["holiday_name"] = join_names

    weekly_features = (
        features_daily.groupby("week_start")
        .agg(weekly_aggregations)
        .reset_index()
    )

    weekly_features = weekly_features.rename(
        columns={
            "date": "week_end",
            "is_holiday": "holiday_days",
            "is_weekend": "weekend_days"
        }
    )

    days_per_week = (
        features_daily.groupby("week_start")["date"]
        .nunique()
        .rename("days_in_period")
        .reset_index()
    )

    weekly_features = weekly_features.merge(
        days_per_week,
        on="week_start",
        how="left",
        validate="one_to_one"
    )

    weekly_features["has_holiday"] = (
        weekly_features["holiday_days"] > 0
    ).astype(int)

    weekly_features["is_partial_week"] = (
        weekly_features["days_in_period"] < 7
    ).astype(int)

    # ------------------------------------------------------
    # 10. INTEGRAR VENTAS SEMANALES Y VARIABLES EXOGENAS
    # ------------------------------------------------------

    weekly = weekly_sales.merge(
        weekly_features,
        on="week_start",
        how="left",
        validate="many_to_one"
    )

    weekly = weekly.sort_values(
        ["shopping_mall", "category", "gender", "week_start"]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # 11. VALIDACIONES FINALES
    # ------------------------------------------------------

    if weekly.empty:
        raise ValueError("El dataset semanal esta vacio.")

    if weekly[target_col].isna().any():
        raise ValueError(
            f"Hay valores faltantes en {target_col}."
        )

    if weekly.duplicated(
        subset=["week_start"] + dimensions
    ).any():
        raise ValueError(
            "Hay semanas duplicadas para una combinacion dimensional."
        )

    # ------------------------------------------------------
    # 12. GUARDAR RESULTADO
    # ------------------------------------------------------

    weekly.to_csv(output_path, index=False)

    # ------------------------------------------------------
    # 13. REPORTE
    # ------------------------------------------------------

    forecast_days = modeling_config.get(
        "forecast_horizon_days", 30
    )

    print("\n========================================")
    print(" DATASET SEMANAL DE MODELADO GENERADO")
    print("========================================")

    print("Registros diarios del panel:", len(daily))
    print("Registros semanales:", len(weekly))
    print("Combinaciones tienda/categoria/genero:",
          len(combinations))

    print("Fecha inicial:", start_date.date())
    print("Fecha final:", end_date.date())

    print("Demanda total:", weekly[target_col].sum())

    if "sales" in weekly.columns:
        print("Ventas totales:", weekly["sales"].sum())

    print("Tiendas:", weekly["shopping_mall"].nunique())
    print("Categorias:", weekly["category"].nunique())
    print("Generos:", weekly["gender"].nunique())

    print("Horizonte de pronostico:", forecast_days, "dias")
    print("Periodos semanales aproximados:",
          (forecast_days + 6) // 7)

    print("Semanas parciales:",
          weekly_features["is_partial_week"].sum())

    print("\nValores faltantes por columna:")
    print(weekly.isna().sum()[weekly.isna().sum() > 0])

    print("\nColumnas generadas:")
    print(weekly.columns.tolist())

    print("\nPrimeros registros:")
    print(weekly.head().to_string(index=False))

    print("\nArchivo generado:", output_path)
    print("\nINTEGRACION COMPLETADA")
    print("========================================")


if __name__ == "__main__":
    merge_datasets()
