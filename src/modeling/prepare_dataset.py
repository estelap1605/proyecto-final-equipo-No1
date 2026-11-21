
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import (
    f_regression,
    mutual_info_regression,
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

CONFIG_PATH = Path("configs/config.yaml")

LAGS = [1, 2, 3, 4, 8, 12]
ROLLING_WINDOWS = [4, 8]

WEATHER_LAGS = [4, 8, 12]

WEATHER_COLUMNS = [
    "temperature_2m_mean",
    "precipitation_sum",
]

N_VALIDATION_FOLDS = 3
HORIZON_WEEKS = 4
RANDOM_STATE = 42

EXPECTED_SERIES = 80

OUTPUT_DIR = Path(
    "data/processed/model_preparation"
)


# ============================================================
# CARGA DE CONFIGURACION
# ============================================================

def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# CARGA DEL DATASET
# ============================================================

def load_modeling_data(config):

    data_path = Path(
        config["data"]["modeling"]["output_path"]
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = [
        "week_start",
        "shopping_mall",
        "category",
        "gender",
        "quantity",
        "holiday_days",
        "weekend_days",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing}"
        )

    # Validar fecha
    df["week_start"] = pd.to_datetime(
        df["week_start"],
        errors="coerce"
    )

    if df["week_start"].isna().any():
        raise ValueError(
            "Existen fechas invalidas en week_start."
        )

    # Validar variable objetivo
    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    if df["quantity"].isna().any():
        raise ValueError(
            "Existen valores invalidos en quantity."
        )

    if (df["quantity"] < 0).any():
        raise ValueError(
            "Existen valores negativos en quantity."
        )

    # Validar variables de calendario
    calendar_columns = [
        "holiday_days",
        "weekend_days",
    ]

    for col in calendar_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if df[col].isna().any():
            raise ValueError(
                f"Existen valores invalidos en {col}."
            )

    # Validar variables meteorologicas
    missing_weather = [
        col for col in WEATHER_COLUMNS
        if col not in df.columns
    ]

    if missing_weather:
        raise ValueError(
            "Faltan variables meteorologicas: "
            f"{missing_weather}"
        )

    for col in WEATHER_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if df[col].isna().any():
            raise ValueError(
                f"Existen valores meteorologicos "
                f"invalidos en {col}."
            )

    return df


# ============================================================
# AGREGACION DE GENEROS
# ============================================================

def aggregate_gender(df):

    df = df.copy()

    group_columns = [
        "week_start",
        "shopping_mall",
        "category",
    ]

    sum_columns = [
        col for col in [
            "quantity",
            "sales",
            "transactions",
        ]
        if col in df.columns
    ]

    excluded_columns = (
        group_columns
        + sum_columns
        + ["gender"]
    )

    other_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    # Verificar que las variables no sumables
    # tengan el mismo valor entre generos.
    for col in other_columns:

        variation = (
            df.groupby(group_columns)[col]
            .nunique(dropna=False)
        )

        if (variation > 1).any():
            raise ValueError(
                f"La columna {col} tiene valores "
                "diferentes entre generos."
            )

    aggregation = {
        col: "sum"
        for col in sum_columns
    }

    aggregation.update({
        col: "first"
        for col in other_columns
    })

    result = (
        df.groupby(
            group_columns,
            as_index=False
        )
        .agg(aggregation)
    )

    return result


# ============================================================
# VALIDACION Y FILTRADO DE SEMANAS
# ============================================================

def validate_and_filter_weeks(df):

    df = df.copy()

    df = df.sort_values(
        [
            "shopping_mall",
            "category",
            "week_start",
        ]
    )

    # Las semanas inicial y final pueden estar incompletas.
    # Se excluyen para trabajar con semanas completas.
    unique_dates = sorted(
        df["week_start"].unique()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "No hay suficientes semanas para preparar "
            "los datasets."
        )

    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    df = df[
        (df["week_start"] > first_date)
        & (df["week_start"] < last_date)
    ].copy()

    # Validar cantidad de series
    series_count = (
        df[["shopping_mall", "category"]]
        .drop_duplicates()
        .shape[0]
    )

    if series_count != EXPECTED_SERIES:
        raise ValueError(
            f"Se esperaban {EXPECTED_SERIES} series, "
            f"pero se encontraron {series_count}."
        )

    # Validar continuidad semanal por serie
    expected_dates = None

    for (mall, category), group in df.groupby(
        ["shopping_mall", "category"]
    ):

        dates = sorted(
            group["week_start"].unique()
        )

        if expected_dates is None:
            expected_dates = dates

        elif dates != expected_dates:
            raise ValueError(
                "Las series no tienen el mismo "
                "periodo semanal."
            )

        date_series = pd.Series(
            pd.to_datetime(dates)
        )

        differences = date_series.diff().dropna()

        if not (differences == pd.Timedelta(days=7)).all():
            raise ValueError(
                f"Existe una discontinuidad semanal "
                f"en {mall} - {category}."
            )

    # Validar registros duplicados
    duplicated = df.duplicated(
        subset=[
            "week_start",
            "shopping_mall",
            "category",
        ]
    )

    if duplicated.any():
        raise ValueError(
            "Existen registros semanales duplicados."
        )

    return df.reset_index(drop=True)


# ============================================================
# VARIABLES DE CALENDARIO
# ============================================================

def add_calendar_features(df):

    df = df.copy()

    required_columns = [
        "holiday_days",
        "weekend_days",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan variables de calendario: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Existen valores nulos en las variables "
            "de calendario."
        )

    dates = pd.to_datetime(df["week_start"])

    week_number = (
        dates.dt.isocalendar().week.astype(int)
    )

    month_number = dates.dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month_number / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month_number / 12
    )

    df["week_sin"] = np.sin(
        2 * np.pi * week_number / 52.18
    )

    df["week_cos"] = np.cos(
        2 * np.pi * week_number / 52.18
    )

    return df


# ============================================================
# CONSTRUCCION DEL DATASET SUPERVISADO
# ============================================================

def build_supervised_dataset(df):

    records = []

    max_lag = max(
        max(LAGS),
        max(WEATHER_LAGS),
        max(ROLLING_WINDOWS),
    )

    group_columns = [
        "shopping_mall",
        "category",
    ]

    for (mall, category), group in df.groupby(
        group_columns
    ):

        group = (
            group.sort_values("week_start")
            .reset_index(drop=True)
        )

        for i in range(max_lag, len(group)):

            row = {
                "week_start": group.loc[i, "week_start"],
                "shopping_mall": mall,
                "category": category,
                "quantity": group.loc[i, "quantity"],
            }

            # ------------------------------------------------
            # Rezagos de demanda
            # ------------------------------------------------

            for lag in LAGS:

                row[f"quantity_lag_{lag}"] = (
                    group.loc[i - lag, "quantity"]
                )

            # ------------------------------------------------
            # Ventanas moviles de demanda
            # Solo utilizan semanas anteriores al objetivo.
            # ------------------------------------------------

            for window in ROLLING_WINDOWS:

                historical_values = (
                    group.loc[
                        i - window:i - 1,
                        "quantity"
                    ]
                )

                row[
                    f"quantity_rolling_mean_{window}"
                ] = historical_values.mean()

                row[
                    f"quantity_rolling_std_{window}"
                ] = historical_values.std(ddof=0)

            # ------------------------------------------------
            # Variables de calendario
            # Son conocidas para las semanas futuras.
            # ------------------------------------------------

            calendar_columns = [
                "holiday_days",
                "weekend_days",
                "month_sin",
                "month_cos",
                "week_sin",
                "week_cos",
            ]

            for col in calendar_columns:

                row[col] = group.loc[i, col]

            # ------------------------------------------------
            # Variables meteorologicas historicas
            # No se utiliza clima observado de la semana
            # objetivo ni de semanas futuras.
            # ------------------------------------------------

            for col in WEATHER_COLUMNS:

                for lag in WEATHER_LAGS:

                    row[f"{col}_lag_{lag}"] = (
                        group.loc[i - lag, col]
                    )

            records.append(row)

    supervised = pd.DataFrame(records)

    if supervised.empty:
        raise ValueError(
            "El dataset supervisado esta vacio."
        )

    supervised = supervised.sort_values(
        [
            "shopping_mall",
            "category",
            "week_start",
        ]
    ).reset_index(drop=True)

    return supervised


# ============================================================
# DIVISION TEMPORAL DE LOS DATOS
# ============================================================

def create_time_splits(df):

    dates = np.array(
        sorted(df["week_start"].unique())
    )

    if len(dates) <= (
        HORIZON_WEEKS * (N_VALIDATION_FOLDS + 1)
    ):
        raise ValueError(
            "No hay suficientes semanas para "
            "crear los folds temporales."
        )

    # Ultimas cuatro semanas para prueba final
    development_dates = dates[:-HORIZON_WEEKS]

    test_dates = dates[-HORIZON_WEEKS:]

    splitter = TimeSeriesSplit(
        n_splits=N_VALIDATION_FOLDS,
        test_size=HORIZON_WEEKS,
    )

    folds = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(development_dates),
        start=1
    ):

        train_dates = development_dates[train_idx]
        validation_dates = development_dates[val_idx]

        folds.append({
            "fold": fold_number,
            "train_dates": train_dates,
            "validation_dates": validation_dates,
        })

    final_test = {
        "train_dates": development_dates,
        "test_dates": test_dates,
    }

    return {
        "development_dates": development_dates,
        "test_dates": test_dates,
        "folds": folds,
        "final_test": final_test,
    }


# ============================================================
# SELECCION DE VARIABLES
# ============================================================

def select_features(
    X,
    y,
    method="mutual_info",
    max_features=10,
):

    X = X.copy()

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # Convertir columnas a formato numerico
    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Eliminar filas sin objetivo valido
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.empty or y.nunique() < 2:
        return []

    # Eliminar columnas completamente vacias
    X = X.dropna(
        axis=1,
        how="all"
    )

    if X.empty:
        return []

    # Imputar valores faltantes con mediana
    X = X.fillna(
        X.median()
    )

    # Eliminar variables constantes
    X = X.loc[
        :,
        X.nunique(dropna=False) > 1
    ]

    if X.empty:
        return []

    # Verificar variabilidad del objetivo
    if y.nunique() < 2:
        return []

    try:

        if method == "f_regression":

            scores, _ = f_regression(
                X,
                y
            )

        elif method == "mutual_info":

            discrete_features = [
                col in [
                    "holiday_days",
                    "weekend_days",
                ]
                for col in X.columns
            ]

            scores = mutual_info_regression(
                X,
                y,
                discrete_features=discrete_features,
                random_state=RANDOM_STATE,
            )

        else:
            raise ValueError(
                f"Metodo de seleccion desconocido: {method}"
            )

    except Exception as error:

        print(
            f"Advertencia en seleccion de variables: {error}"
        )

        return []

    scores = np.asarray(
        scores,
        dtype=float
    )

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    score_df = pd.DataFrame({
        "feature": X.columns,
        "score": scores,
    })

    score_df = score_df[
        score_df["score"] > 0
    ]

    if score_df.empty:
        return []

    score_df = score_df.sort_values(
        "score",
        ascending=False
    )

    selected = score_df.head(
        max_features
    )["feature"].tolist()

    return selected


# ============================================================
# SELECCION DE VARIABLES POR MODELO, FOLD Y SERIE
# ============================================================

def perform_feature_selection(
    supervised,
    splits
):

    weather_features = [
        f"{col}_lag_{lag}"
        for col in WEATHER_COLUMNS
        for lag in WEATHER_LAGS
    ]

    calendar_features = [
        "holiday_days",
        "weekend_days",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
    ]

    demand_features = [
        col for col in supervised.columns
        if (
            col.startswith("quantity_lag_")
            or col.startswith("quantity_rolling_")
        )
    ]

    all_predictors = (
        demand_features
        + calendar_features
        + weather_features
    )

    model_specs = {
        "linear_regression": {
            "method": "f_regression",
            "max_features": 8,
            "features": all_predictors,
        },

        "lstm": {
            "method": "mutual_info",
            "max_features": 10,
            "features": all_predictors,
        },

        "sarimax": {
            "method": "mutual_info",
            "max_features": 4,
            "features": (
                calendar_features
                + weather_features
            ),
        },
    }

    selection_results = {}

    partitions = []

    for fold in splits["folds"]:

        partitions.append({
            "name": f"fold_{fold['fold']}",
            "train_dates": fold["train_dates"],
        })

    partitions.append({
        "name": "final_test",
        "train_dates": splits["final_test"]["train_dates"],
    })

    for partition in partitions:

        partition_name = partition["name"]

        train_dates = pd.to_datetime(
            partition["train_dates"]
        )

        selection_results[partition_name] = {}

        for model_name, spec in model_specs.items():

            selection_results[partition_name][
                model_name
            ] = {}

            candidate_features = [
                col for col in spec["features"]
                if col in supervised.columns
            ]

            for (mall, category), group in supervised.groupby(
                ["shopping_mall", "category"]
            ):

                train_data = group[
                    group["week_start"].isin(train_dates)
                ].copy()

                if train_data.empty:

                    selected = []

                else:

                    X = train_data[
                        candidate_features
                    ]

                    y = train_data["quantity"]

                    selected = select_features(
                        X=X,
                        y=y,
                        method=spec["method"],
                        max_features=spec["max_features"],
                    )

                series_name = (
                    f"{mall} | {category}"
                )

                selection_results[
                    partition_name
                ][
                    model_name
                ][
                    series_name
                ] = selected

    return selection_results


# ============================================================
# METADATOS DE LOS SPLITS
# ============================================================

def build_split_metadata(splits):

    metadata = {
        "development_dates": [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in splits["development_dates"]
        ],

        "test_dates": [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in splits["test_dates"]
        ],

        "folds": [],
    }

    for fold in splits["folds"]:

        metadata["folds"].append({
            "fold": fold["fold"],

            "train_weeks": len(
                fold["train_dates"]
            ),

            "validation_weeks": len(
                fold["validation_dates"]
            ),

            "train_start": pd.Timestamp(
                fold["train_dates"][0]
            ).strftime("%Y-%m-%d"),

            "train_end": pd.Timestamp(
                fold["train_dates"][-1]
            ).strftime("%Y-%m-%d"),

            "validation_start": pd.Timestamp(
                fold["validation_dates"][0]
            ).strftime("%Y-%m-%d"),

            "validation_end": pd.Timestamp(
                fold["validation_dates"][-1]
            ).strftime("%Y-%m-%d"),
        })

    metadata["final_test"] = {
        "train_weeks": len(
            splits["final_test"]["train_dates"]
        ),

        "test_weeks": len(
            splits["final_test"]["test_dates"]
        ),

        "test_start": pd.Timestamp(
            splits["final_test"]["test_dates"][0]
        ).strftime("%Y-%m-%d"),

        "test_end": pd.Timestamp(
            splits["final_test"]["test_dates"][-1]
        ).strftime("%Y-%m-%d"),
    }

    return metadata


# ============================================================
# CONSTRUCCION DEL ARCHIVO DE FECHAS DE LOS SPLITS
# ============================================================

def build_split_dates(splits):

    records = []

    for fold in splits["folds"]:

        fold_name = fold["fold"]

        for date in fold["train_dates"]:

            records.append({
                "fold": fold_name,
                "date": date,
                "partition": "train",
            })

        for date in fold["validation_dates"]:

            records.append({
                "fold": fold_name,
                "date": date,
                "partition": "validation",
            })

    # Prueba final
    for date in splits["final_test"]["train_dates"]:

        records.append({
            "fold": "final_test",
            "date": date,
            "partition": "train",
        })

    for date in splits["final_test"]["test_dates"]:

        records.append({
            "fold": "final_test",
            "date": date,
            "partition": "test",
        })

    return pd.DataFrame(records)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_outputs(
    series_df,
    supervised,
    splits,
    feature_selection
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Dataset por serie
    series_df.to_csv(
        OUTPUT_DIR / "series_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Dataset supervisado
    supervised.to_csv(
        OUTPUT_DIR / "supervised_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Metadatos
    metadata = build_split_metadata(
        splits
    )

    with open(
        OUTPUT_DIR / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Seleccion de variables
    with open(
        OUTPUT_DIR / "feature_selection.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_selection,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Fechas de los splits
    split_dates = build_split_dates(
        splits
    )

    split_dates.to_csv(
        OUTPUT_DIR / "split_dates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArchivos guardados en:")
    print(OUTPUT_DIR)


# ============================================================
# RESUMEN DE SELECCION DE VARIABLES
# ============================================================

def print_feature_selection_summary(
    feature_selection
):

    print("\nRESUMEN DE SELECCION DE VARIABLES")
    print("----------------------------------------")

    for partition, models in feature_selection.items():

        print(f"\n{partition}")

        for model_name, series_results in models.items():

            counts = [
                len(features)
                for features in series_results.values()
            ]

            empty_series = sum(
                count == 0
                for count in counts
            )

            average_features = (
                np.mean(counts)
                if counts else 0
            )

            print(
                f"{model_name}: "
                f"{len(counts)} series, "
                f"{empty_series} sin variables, "
                f"promedio {average_features:.2f}"
            )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    print("\n========================================")
    print(" PREPARACION DE DATASETS PARA MODELADO")
    print("========================================")

    # 1. Configuracion
    config = load_config()

    # 2. Cargar datos
    df = load_modeling_data(config)

    print(
        f"\nRegistros originales: {len(df)}"
    )

    # 3. Agregar generos
    series_df = aggregate_gender(df)

    series_count = (
        series_df[
            ["shopping_mall", "category"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Series despues de agregar generos: "
        f"{series_count}"
    )

    # 4. Validar semanas completas
    series_df = validate_and_filter_weeks(
        series_df
    )

    print(
        f"Registros semanales completos: "
        f"{len(series_df)}"
    )

    weeks_per_series = (
        series_df.groupby(
            ["shopping_mall", "category"]
        )["week_start"]
        .nunique()
    )

    print(
        f"Semanas completas por serie: "
        f"{weeks_per_series.iloc[0]}"
    )

    # 5. Variables de calendario
    series_df = add_calendar_features(
        series_df
    )

    # 6. Dataset supervisado
    supervised = build_supervised_dataset(
        series_df
    )

    print(
        f"Registros del dataset supervisado: "
        f"{len(supervised)}"
    )

    # 7. Validar variables meteorologicas
    weather_features = [
        col for col in supervised.columns
        if (
            "temperature" in col
            or "precipitation" in col
        )
    ]

    print(
        f"Variables meteorologicas creadas: "
        f"{len(weather_features)}"
    )

    for col in weather_features:

        if supervised[col].isna().any():
            raise ValueError(
                f"Existen nulos en {col}."
            )

    # 8. Crear splits temporales
    splits = create_time_splits(
        series_df
    )

    print("\n----------------------------------------")
    print("RESUMEN DE SPLITS")
    print("----------------------------------------")

    for fold in splits["folds"]:

        print(
            f"Fold {fold['fold']}: "
            f"{len(fold['train_dates'])} semanas train, "
            f"{len(fold['validation_dates'])} validation"
        )

    print(
        f"Prueba final: "
        f"{len(splits['final_test']['train_dates'])} "
        f"semanas de entrenamiento y "
        f"{len(splits['final_test']['test_dates'])} "
        f"de prueba"
    )

    # 9. Seleccion de variables
    feature_selection = perform_feature_selection(
        supervised,
        splits
    )

    # 10. Guardar archivos
    save_outputs(
        series_df=series_df,
        supervised=supervised,
        splits=splits,
        feature_selection=feature_selection,
    )

    # 11. Resumen
    print_feature_selection_summary(
        feature_selection
    )

    print("\n========================================")
    print(" PREPARACION COMPLETADA")
    print("========================================")


if __name__ == "__main__":
    main()
