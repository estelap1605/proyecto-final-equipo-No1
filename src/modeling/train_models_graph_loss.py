
"""
Entrenamiento y evaluación de Regresión Lineal, SARIMAX y LSTM.
Experimento B: demanda histórica + calendario + clima.

Ejecución desde la raíz del proyecto:
    python -m src.modeling.train_models

Entradas:
    data/processed/model_preparation/series_dataset.csv
    data/processed/model_preparation/supervised_dataset.csv
    data/processed/model_preparation/feature_selection.json
    data/processed/model_preparation/split_metadata.json

Salidas:
    data/processed/model_results/
        predictions/
        metrics/
        models/
        loss/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class ModelTrainer:
    """Entrena y evalúa tres modelos con pronósticos de origen fijo."""

    def __init__(
        self,
        preparation_dir="data/processed/model_preparation",
        output_dir="data/processed/model_results",
        horizon=4,
        lstm_lookback=8,
        random_state=42,
        lstm_epochs=60,
        lstm_batch_size=16,
    ):
        self.preparation_dir = Path(preparation_dir)
        self.output_dir = Path(output_dir)

        self.horizon = horizon
        self.lstm_lookback = lstm_lookback
        self.random_state = random_state
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size

        self.series_path = self.preparation_dir / "series_dataset.csv"
        self.supervised_path = self.preparation_dir / "supervised_dataset.csv"
        self.selection_path = self.preparation_dir / "feature_selection.json"
        self.splits_path = self.preparation_dir / "split_metadata.json"

        self.series = None
        self.supervised = None
        self.feature_selection = None
        self.split_metadata = None

        self.loss_dir = self.output_dir / "loss"

        for folder in [
            self.output_dir,
            self.output_dir / "predictions",
            self.output_dir / "metrics",
            self.output_dir / "models",
            self.loss_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    # ==============================================================
    # CARGA DE DATOS
    # ==============================================================

    def load_data(self):
        required = [
            self.series_path,
            self.supervised_path,
            self.selection_path,
            self.splits_path,
        ]

        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "No se encontraron los siguientes archivos:\n"
                + "\n".join(missing)
            )

        self.series = pd.read_csv(
            self.series_path,
            parse_dates=["week_start"],
        )

        self.supervised = pd.read_csv(
            self.supervised_path,
            parse_dates=["week_start"],
        )

        self.feature_selection = json.loads(
            self.selection_path.read_text(encoding="utf-8")
        )

        self.split_metadata = json.loads(
            self.splits_path.read_text(encoding="utf-8")
        )

        required_columns = {
            "week_start",
            "shopping_mall",
            "category",
            "quantity",
        }

        for name, frame in [
            ("series_dataset.csv", self.series),
            ("supervised_dataset.csv", self.supervised),
        ]:
            missing_columns = required_columns.difference(frame.columns)

            if missing_columns:
                raise ValueError(
                    f"{name}: faltan columnas "
                    f"{sorted(missing_columns)}"
                )

        self.series = (
            self.series
            .sort_values(
                ["shopping_mall", "category", "week_start"]
            )
            .reset_index(drop=True)
        )

        self.supervised = (
            self.supervised
            .sort_values(
                ["shopping_mall", "category", "week_start"]
            )
            .reset_index(drop=True)
        )

        number_series = (
            self.series[
                ["shopping_mall", "category"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        if number_series != 80:
            raise ValueError(
                f"Se esperaban 80 series; se encontraron {number_series}."
            )

        print(f"Series cargadas: {number_series}")
        print(f"Registros históricos: {len(self.series):,}")
        print(f"Registros supervisados: {len(self.supervised):,}")

        return self

    # ==============================================================
    # PARTICIONES
    # ==============================================================

    @staticmethod
    def _date_list(values):
        return pd.to_datetime(values).tolist()

    def _get_partitions(self):
        meta = self.split_metadata
        partitions = []

        folds = meta.get("folds", [])

        if isinstance(folds, dict):
            folds = list(folds.values())

        for i, fold in enumerate(folds, start=1):

            train_dates = fold.get(
                "train_dates",
                fold.get("train", []),
            )

            validation_dates = fold.get(
                "validation_dates",
                fold.get(
                    "val_dates",
                    fold.get("validation", []),
                ),
            )

            if train_dates and validation_dates:
                partitions.append(
                    (
                        f"fold_{fold.get('fold', i)}",
                        self._date_list(train_dates),
                        self._date_list(validation_dates),
                    )
                )

        final = meta.get("final_test", {})

        if final:
            train_dates = final.get(
                "train_dates",
                meta.get("development_dates", []),
            )

            test_dates = final.get(
                "test_dates",
                meta.get("test_dates", []),
            )
        else:
            train_dates = meta.get("development_dates", [])
            test_dates = meta.get("test_dates", [])

        if train_dates and test_dates:
            partitions.append(
                (
                    "final_test",
                    self._date_list(train_dates),
                    self._date_list(test_dates),
                )
            )

        if not partitions:
            raise ValueError(
                "No fue posible interpretar los folds y el test. "
                "Revisa split_metadata.json."
            )

        return partitions

    # ==============================================================
    # SELECCIÓN DE VARIABLES
    # ==============================================================

    @staticmethod
    def _series_key(mall, category):
        return f"{mall} | {category}"

    def _selected_features(
        self,
        partition,
        model_name,
        mall,
        category,
        candidates,
    ):
        node = self.feature_selection

        for key in (
            partition,
            model_name,
            self._series_key(mall, category),
        ):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break

        if isinstance(node, list):
            selected = node

        elif isinstance(node, dict):
            selected = node.get(
                "selected_features",
                node.get("features", []),
            )

        else:
            selected = []

        selected = [
            column
            for column in selected
            if column in candidates
        ]

        # Si no hay variables seleccionadas, se utilizan las candidatas.
        if not selected:
            selected = list(candidates)

        if not selected:
            raise ValueError(
                f"No hay variables predictoras para "
                f"{mall} | {category} | {model_name}."
            )

        return selected

    @staticmethod
    def _feature_columns(frame):
        excluded = {
            "week_start",
            "shopping_mall",
            "category",
            "quantity",
        }

        return [
            column
            for column in frame.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

    # ==============================================================
    # LIMPIEZA DE VARIABLES
    # ==============================================================

    @staticmethod
    def _clean_matrix(train_x, future_x):
        """
        Calcula medianas exclusivamente con entrenamiento.
        Las columnas completamente vacías reciben valor 0.
        """

        train_x = train_x.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        future_x = future_x.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        medians = train_x.median()
        medians = medians.fillna(0.0)

        train_clean = train_x.fillna(medians)
        future_clean = future_x.fillna(medians)

        return (
            train_clean.to_numpy(dtype=float),
            future_clean.to_numpy(dtype=float),
            medians,
        )

    @staticmethod
    def _apply_medians(frame, medians):
        frame = frame.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return frame.fillna(medians).to_numpy(dtype=float)

    # ==============================================================
    # MÉTRICAS
    # ==============================================================

    @staticmethod
    def calculate_metrics(y_true, y_pred):

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        valid = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
        )

        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            return {
                "WAPE": np.nan,
                "MAPE": np.nan,
                "RMSE": np.nan,
                "mape_zero_excluded": 0,
                "n_observations": 0,
            }

        denominator = np.abs(y_true).sum()

        wape = (
            100.0
            * np.abs(y_true - y_pred).sum()
            / denominator
            if denominator > 0
            else np.nan
        )

        nonzero = y_true != 0

        mape = (
            100.0
            * np.mean(
                np.abs(
                    (y_true[nonzero] - y_pred[nonzero])
                    / y_true[nonzero]
                )
            )
            if nonzero.any()
            else np.nan
        )

        rmse = float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        )

        return {
            "WAPE": float(wape) if np.isfinite(wape) else np.nan,
            "MAPE": float(mape) if np.isfinite(mape) else np.nan,
            "RMSE": rmse,
            "mape_zero_excluded": int((~nonzero).sum()),
            "n_observations": int(len(y_true)),
        }

    # ==============================================================
    # ACTUALIZACIÓN RECURSIVA DE VARIABLES
    # ==============================================================

    def _make_future_row(
        self,
        history,
        template,
        feature_columns,
    ):
        """
        Actualiza rezagos y ventanas móviles de demanda.
        Los pasos futuros utilizan predicciones anteriores.
        """

        row = template.copy()
        values = np.asarray(history, dtype=float)

        for column in feature_columns:

            if column.startswith("quantity_lag_"):
                lag = int(column.rsplit("_", 1)[1])

                row[column] = (
                    values[-lag]
                    if len(values) >= lag
                    else np.nan
                )

            elif column.startswith("quantity_rolling_mean_"):
                window = int(column.rsplit("_", 1)[1])

                row[column] = (
                    np.mean(values[-window:])
                    if len(values) >= window
                    else np.nan
                )

            elif column.startswith("quantity_rolling_std_"):
                window = int(column.rsplit("_", 1)[1])

                row[column] = (
                    np.std(values[-window:], ddof=0)
                    if len(values) >= window
                    else np.nan
                )

        return row

    # ==============================================================
    # REGRESIÓN LINEAL
    # ==============================================================

    def train_linear_regression(
        self,
        train_frame,
        future_templates,
        features,
    ):

        y = train_frame["quantity"].astype(float).to_numpy()

        split = max(
            1,
            int(len(train_frame) * 0.8),
        )

        if len(train_frame) - split < 2:
            raise ValueError(
                "No hay suficientes observaciones para "
                "validación interna de regresión lineal."
            )

        # Diagnóstico de entrenamiento y validación temporal.
        X_fit = train_frame.iloc[:split][features].copy()
        y_fit = y[:split]

        X_val = train_frame.iloc[split:][features].copy()
        y_val = y[split:]

        X_fit_clean, X_val_clean, _ = self._clean_matrix(
            X_fit,
            X_val,
        )

        diagnostic_model = LinearRegression()
        diagnostic_model.fit(
            X_fit_clean,
            y_fit,
        )

        train_mse = mean_squared_error(
            y_fit,
            diagnostic_model.predict(X_fit_clean),
        )

        validation_mse = mean_squared_error(
            y_val,
            diagnostic_model.predict(X_val_clean),
        )

        # Modelo definitivo con todo el entrenamiento externo.
        X_train = train_frame[features].copy()
        X_future = future_templates[features].copy()

        X_train_clean, _, medians = self._clean_matrix(
            X_train,
            X_future,
        )

        model = LinearRegression()
        model.fit(X_train_clean, y)

        history = y.tolist()
        predictions = []

        for step in range(len(future_templates)):

            template = future_templates.iloc[step].copy()

            dynamic = self._make_future_row(
                history,
                template,
                features,
            )

            x = pd.DataFrame(
                [dynamic]
            )[features]

            x_clean = self._apply_medians(
                x,
                medians,
            )

            prediction = max(
                0.0,
                float(model.predict(x_clean)[0]),
            )

            predictions.append(prediction)
            history.append(prediction)

        return predictions, model, {
            "train_loss": [float(train_mse)],
            "validation_loss": [float(validation_mse)],
        }

    # ==============================================================
    # SARIMAX
    # ==============================================================

    def train_sarimax(
        self,
        train_frame,
        future_templates,
        features,
    ):

        y_train = train_frame["quantity"].astype(float).to_numpy()

        X_train = train_frame[features].copy()

        X_future = future_templates[features].copy()

        X_train_clean, _, medians = self._clean_matrix(
            X_train,
            X_future,
        )

        model = SARIMAX(
            y_train,
            exog=X_train_clean,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        optimizer_loss = []

        def record_optimizer_loss(parameters):
            try:
                value = float(
                    -model.loglike(parameters)
                )

                if np.isfinite(value):
                    optimizer_loss.append(value)

            except Exception:
                pass

        fitted = model.fit(
            disp=False,
            maxiter=200,
            callback=record_optimizer_loss,
        )

        if not optimizer_loss:
            optimizer_loss = [
                float(-fitted.llf)
            ]

        # Pronóstico recursivo.
        history = y_train.tolist()
        current_result = fitted
        predictions = []

        for step in range(len(future_templates)):

            template = future_templates.iloc[step].copy()

            dynamic = self._make_future_row(
                history,
                template,
                features,
            )

            X_step = pd.DataFrame(
                [dynamic]
            )[features]

            X_step_clean = self._apply_medians(
                X_step,
                medians,
            )

            forecast = current_result.get_forecast(
                steps=1,
                exog=X_step_clean,
            )

            prediction = max(
                0.0,
                float(
                    np.asarray(
                        forecast.predicted_mean
                    ).ravel()[0]
                ),
            )

            predictions.append(prediction)
            history.append(prediction)

            # Actualiza el estado con la predicción, no con
            # la demanda real de la semana futura.
            current_result = current_result.append(
                endog=[prediction],
                exog=X_step_clean,
                refit=False,
            )

        return predictions, fitted, {
            "values": optimizer_loss,
        }

    # ==============================================================
    # ARQUITECTURA LSTM
    # ==============================================================

    @staticmethod
    def _build_lstm_model(lookback, number_features):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            LSTM,
            Dense,
            Dropout,
        )
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(
                32,
                input_shape=(
                    lookback,
                    number_features,
                ),
                return_sequences=False,
            ),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
        )

        return model

    # ==============================================================
    # LSTM
    # ==============================================================

    def train_lstm(
        self,
        train_frame,
        future_templates,
        features,
    ):

        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping

        except ImportError as exc:
            raise ImportError(
                "TensorFlow no está instalado en el entorno activo. "
                "Instálalo en Python 3.12."
            ) from exc

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        X_raw = train_frame[features].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        y_raw = (
            train_frame["quantity"]
            .astype(float)
            .to_numpy()
            .reshape(-1, 1)
        )

        lookback = min(
            self.lstm_lookback,
            len(train_frame) - 5,
        )

        if lookback < 2:
            raise ValueError(
                "No hay suficientes observaciones para LSTM."
            )

        # Construcción de secuencias.
        number_sequences = len(train_frame) - lookback

        split = max(
            1,
            int(number_sequences * 0.8),
        )

        if number_sequences - split < 1:
            raise ValueError(
                "No hay suficientes secuencias para validación LSTM."
            )

        # Los objetivos de entrenamiento terminan antes del límite.
        raw_boundary = lookback + split

        # ----------------------------------------------------------
        # FASE 1: entrenamiento y validación temporal
        # ----------------------------------------------------------

        # Los escaladores y el imputador solo conocen el segmento
        # de entrenamiento interno.
        X_internal_train = X_raw.iloc[:raw_boundary]

        imputer_internal = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

        X_internal_clean = imputer_internal.fit_transform(
            X_internal_train
        )

        X_internal_all = imputer_internal.transform(X_raw)

        x_scaler_internal = StandardScaler()

        x_scaler_internal.fit(X_internal_clean)

        X_internal_scaled = x_scaler_internal.transform(
            X_internal_all
        )

        y_scaler_internal = MinMaxScaler()

        y_scaler_internal.fit(
            y_raw[:raw_boundary]
        )

        y_internal_scaled = (
            y_scaler_internal
            .transform(y_raw)
            .ravel()
        )

        X_sequences = []
        y_sequences = []

        for i in range(lookback, len(X_internal_scaled)):

            X_sequences.append(
                X_internal_scaled[i - lookback:i]
            )

            y_sequences.append(
                y_internal_scaled[i]
            )

        X_sequences = np.asarray(
            X_sequences,
            dtype=np.float32,
        )

        y_sequences = np.asarray(
            y_sequences,
            dtype=np.float32,
        )

        X_train_seq = X_sequences[:split]
        y_train_seq = y_sequences[:split]

        X_val_seq = X_sequences[split:]
        y_val_seq = y_sequences[split:]

        diagnostic_model = self._build_lstm_model(
            lookback,
            len(features),
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        )

        diagnostic_history = diagnostic_model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(
                X_val_seq,
                y_val_seq,
            ),
            epochs=self.lstm_epochs,
            batch_size=min(
                self.lstm_batch_size,
                len(X_train_seq),
            ),
            verbose=0,
            callbacks=[early_stop],
            shuffle=False,
        )

        train_loss = [
            float(value)
            for value in diagnostic_history.history["loss"]
        ]

        validation_loss = [
            float(value)
            for value in diagnostic_history.history["val_loss"]
        ]

        # Número de épocas con menor val_loss.
        best_epoch = int(
            np.argmin(validation_loss) + 1
        )

        # ----------------------------------------------------------
        # FASE 2: modelo definitivo con todo el entrenamiento externo
        # ----------------------------------------------------------

        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

        X_clean = imputer.fit_transform(X_raw)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X_clean)

        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y_raw).ravel()

        X_final_sequences = []
        y_final_sequences = []

        for i in range(lookback, len(X_scaled)):

            X_final_sequences.append(
                X_scaled[i - lookback:i]
            )

            y_final_sequences.append(
                y_scaled[i]
            )

        X_final_sequences = np.asarray(
            X_final_sequences,
            dtype=np.float32,
        )

        y_final_sequences = np.asarray(
            y_final_sequences,
            dtype=np.float32,
        )

        final_model = self._build_lstm_model(
            lookback,
            len(features),
        )

        final_model.fit(
            X_final_sequences,
            y_final_sequences,
            epochs=best_epoch,
            batch_size=min(
                self.lstm_batch_size,
                len(X_final_sequences),
            ),
            verbose=0,
            shuffle=False,
        )

        # ----------------------------------------------------------
        # PRONÓSTICO RECURSIVO
        # ----------------------------------------------------------

        history = (
            train_frame["quantity"]
            .astype(float)
            .tolist()
        )

        feature_history = [
            row.copy()
            for row in X_scaled
        ]

        predictions = []

        for step in range(len(future_templates)):

            template = future_templates.iloc[step].copy()

            dynamic = self._make_future_row(
                history,
                template,
                features,
            )

            row_df = pd.DataFrame(
                [dynamic]
            )[features].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            row_clean = imputer.transform(row_df)

            row_scaled = x_scaler.transform(
                row_clean
            )[0]

            sequence_history = (
                feature_history + [row_scaled]
            )

            sequence = np.asarray(
                sequence_history[-lookback:],
                dtype=np.float32,
            )

            if len(sequence) < lookback:
                padding = np.repeat(
                    sequence[:1],
                    lookback - len(sequence),
                    axis=0,
                )

                sequence = np.vstack([
                    padding,
                    sequence,
                ])

            prediction_scaled = float(
                final_model.predict(
                    sequence[None, :, :],
                    verbose=0,
                )[0, 0]
            )

            prediction = float(
                y_scaler.inverse_transform(
                    [[prediction_scaled]]
                )[0, 0]
            )

            prediction = max(0.0, prediction)

            predictions.append(prediction)
            history.append(prediction)
            feature_history.append(row_scaled)

        return predictions, final_model, {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_epoch": best_epoch,
        }

    # ==============================================================
    # NOMBRES SEGUROS PARA ARCHIVOS
    # ==============================================================

    @staticmethod
    def _safe_name(value):
        return "".join(
            character
            if character.isalnum() or character in "-_."
            else "_"
            for character in str(value)
        )

    # ==============================================================
    # GRÁFICAS DE LOSS
    # ==============================================================

    def _plot_loss_curves(
        self,
        partition,
        diagnostics,
    ):

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        for model_name, records in diagnostics.items():

            if not records:
                continue

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            # ------------------------------------------------------
            # REGRESIÓN LINEAL
            # ------------------------------------------------------

            if model_name == "linear_regression":

                usable = [
                    record
                    for record in records
                    if record.get("train_loss")
                    and record.get("validation_loss")
                ]

                if not usable:
                    plt.close(fig)
                    continue

                labels = [
                    record["series"]
                    for record in usable
                ]

                train_values = [
                    record["train_loss"][0]
                    for record in usable
                ]

                validation_values = [
                    record["validation_loss"][0]
                    for record in usable
                ]

                x = np.arange(len(labels))
                width = 0.40

                ax.bar(
                    x - width / 2,
                    train_values,
                    width,
                    label="MSE entrenamiento",
                )

                ax.bar(
                    x + width / 2,
                    validation_values,
                    width,
                    label="MSE validación",
                )

                ax.set_xticks(x)

                ax.set_xticklabels(
                    labels,
                    rotation=90,
                    fontsize=6,
                )

                ax.set_xlabel(
                    "Centro comercial | categoría"
                )

                ax.set_ylabel("MSE")

                ax.set_title(
                    f"Regresión Lineal — entrenamiento vs validación — {partition}"
                )

                ax.legend()

            # ------------------------------------------------------
            # SARIMAX
            # ------------------------------------------------------

            elif model_name == "sarimax":

                usable = [
                    record
                    for record in records
                    if record.get("values")
                ]

                if not usable:
                    plt.close(fig)
                    continue

                plotted = 0

                for record in usable:

                    values = np.asarray(
                        record["values"],
                        dtype=float,
                    )

                    values = values[
                        np.isfinite(values)
                    ]

                    if len(values) == 0:
                        continue

                    label = (
                        record["series"]
                        if plotted < 5
                        else None
                    )

                    ax.plot(
                        np.arange(1, len(values) + 1),
                        values,
                        alpha=0.30,
                        linewidth=0.9,
                        label=label,
                    )

                    plotted += 1

                ax.set_xlabel(
                    "Iteración del optimizador"
                )

                ax.set_ylabel(
                    "−Log-verosimilitud"
                )

                ax.set_title(
                    f"SARIMAX — optimización por serie — {partition}"
                )

                if plotted:
                    ax.legend(
                        fontsize=7,
                        title="Primeras cinco series",
                    )

            # ------------------------------------------------------
            # LSTM
            # ------------------------------------------------------

            else:

                usable = [
                    record
                    for record in records
                    if record.get("train_loss")
                    and record.get("validation_loss")
                ]

                if not usable:
                    plt.close(fig)
                    continue

                max_epochs = max(
                    len(record["train_loss"])
                    for record in usable
                )

                train_matrix = np.full(
                    (len(usable), max_epochs),
                    np.nan,
                )

                validation_matrix = np.full(
                    (len(usable), max_epochs),
                    np.nan,
                )

                for i, record in enumerate(usable):

                    train_values = np.asarray(
                        record["train_loss"],
                        dtype=float,
                    )

                    validation_values = np.asarray(
                        record["validation_loss"],
                        dtype=float,
                    )

                    train_matrix[
                        i,
                        :len(train_values)
                    ] = train_values

                    validation_matrix[
                        i,
                        :len(validation_values)
                    ] = validation_values

                train_mean = np.nanmean(
                    train_matrix,
                    axis=0,
                )

                validation_mean = np.nanmean(
                    validation_matrix,
                    axis=0,
                )

                epochs = np.arange(
                    1,
                    max_epochs + 1,
                )

                ax.plot(
                    epochs,
                    train_mean,
                    label="Loss entrenamiento",
                )

                ax.plot(
                    epochs,
                    validation_mean,
                    label="Loss validación",
                )

                ax.set_xlabel("Época")
                ax.set_ylabel("MSE")

                ax.set_title(
                    f"LSTM — entrenamiento y validación — {partition}"
                )

                ax.legend()

            ax.grid(
                True,
                alpha=0.3,
            )

            fig.tight_layout()

            output = (
                self.loss_dir
                / f"{partition}_{model_name}_loss.png"
            )

            fig.savefig(
                output,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"  Gráfica de loss: {output}"
            )

    # ==============================================================
    # EVALUACIÓN DE UNA PARTICIÓN
    # ==============================================================

    def evaluate_partition(
        self,
        partition,
        train_dates,
        eval_dates,
    ):

        all_predictions = []
        series_metrics = []

        diagnostics = {
            "linear_regression": [],
            "sarimax": [],
            "lstm": [],
        }

        models = {
            "linear_regression": self.train_linear_regression,
            "sarimax": self.train_sarimax,
            "lstm": self.train_lstm,
        }

        train_dates = pd.to_datetime(train_dates)
        eval_dates = pd.to_datetime(eval_dates)

        for (mall, category), full_series in self.series.groupby(
            ["shopping_mall", "category"],
            sort=True,
        ):

            sup = self.supervised[
                (self.supervised["shopping_mall"] == mall)
                & (self.supervised["category"] == category)
            ].sort_values(
                "week_start"
            ).reset_index(drop=True)

            train = sup[
                sup["week_start"].isin(train_dates)
            ].copy()

            future = sup[
                sup["week_start"].isin(eval_dates)
            ].copy()

            future = (
                future
                .sort_values("week_start")
                .reset_index(drop=True)
            )

            if len(future) != len(eval_dates):
                raise ValueError(
                    f"{mall} | {category}: faltan filas futuras "
                    f"en {partition}."
                )

            if train.empty:
                raise ValueError(
                    f"{mall} | {category}: entrenamiento vacío."
                )

            if train["week_start"].max() >= future["week_start"].min():
                raise ValueError(
                    "Se detectó solapamiento temporal entre "
                    "entrenamiento y evaluación."
                )

            candidates = self._feature_columns(train)

            for model_name, trainer in models.items():

                features = self._selected_features(
                    partition,
                    model_name,
                    mall,
                    category,
                    candidates,
                )

                status = "ok"
                error = ""

                try:

                    preds, fitted, loss_info = trainer(
                        train,
                        future,
                        features,
                    )

                    if len(preds) != len(future):
                        raise ValueError(
                            "El modelo no generó el número esperado "
                            "de predicciones."
                        )

                    diagnostics[model_name].append({
                        "series": self._series_key(
                            mall,
                            category,
                        ),
                        **loss_info,
                    })

                    # Guardar modelo.
                    model_path = (
                        self.output_dir
                        / "models"
                        / partition
                        / model_name
                        / (
                            f"{self._safe_name(mall)}"
                            f"__{self._safe_name(category)}"
                        )
                    )

                    model_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    if model_name == "lstm":

                        fitted.save(
                            model_path.with_suffix(".keras")
                        )

                    else:

                        import joblib

                        joblib.dump(
                            fitted,
                            model_path.with_suffix(".pkl"),
                        )

                except Exception as exc:

                    preds = [
                        np.nan
                    ] * len(future)

                    status = "error"
                    error = str(exc)

                    warnings.warn(
                        f"Error {partition} / {model_name} / "
                        f"{mall} / {category}: {exc}"
                    )

                # Guardar predicciones.
                for date, actual, pred in zip(
                    future["week_start"],
                    future["quantity"],
                    preds,
                ):

                    all_predictions.append({
                        "partition": partition,
                        "model": model_name,
                        "shopping_mall": mall,
                        "category": category,
                        "week_start": date,
                        "y_true": float(actual),
                        "y_pred": (
                            float(pred)
                            if np.isfinite(pred)
                            else np.nan
                        ),
                        "status": status,
                        "error": error,
                    })

                valid_preds = np.isfinite(
                    np.asarray(preds, dtype=float)
                )

                if valid_preds.any():

                    metrics = self.calculate_metrics(
                        future.loc[
                            valid_preds,
                            "quantity",
                        ].to_numpy(),
                        np.asarray(
                            preds,
                            dtype=float,
                        )[valid_preds],
                    )

                else:

                    metrics = self.calculate_metrics(
                        [],
                        [],
                    )

                series_metrics.append({
                    "partition": partition,
                    "model": model_name,
                    "shopping_mall": mall,
                    "category": category,
                    **metrics,
                    "status": status,
                    "error": error,
                })

        # Generar gráficas.
        self._plot_loss_curves(
            partition,
            diagnostics,
        )

        predictions = pd.DataFrame(
            all_predictions
        )

        metrics_by_series = pd.DataFrame(
            series_metrics
        )

        # ----------------------------------------------------------
        # MÉTRICAS MACRO: PROMEDIO ENTRE SERIES
        # ----------------------------------------------------------

        macro = (
            metrics_by_series
            .groupby(
                ["partition", "model"],
                dropna=False,
            )
            .agg(
                WAPE_mean_series=("WAPE", "mean"),
                MAPE_mean_series=("MAPE", "mean"),
                RMSE_mean_series=("RMSE", "mean"),
                successful_series=(
                    "status",
                    lambda x: int((x == "ok").sum()),
                ),
                total_series=("status", "size"),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # MÉTRICAS GLOBALES
        # ----------------------------------------------------------

        global_rows = []

        for (part, model_name), group in predictions.groupby(
            ["partition", "model"]
        ):

            valid = group[
                ["y_true", "y_pred"]
            ].notna().all(axis=1)

            metric = self.calculate_metrics(
                group.loc[valid, "y_true"],
                group.loc[valid, "y_pred"],
            )

            global_rows.append({
                "partition": part,
                "model": model_name,
                "WAPE_global": metric["WAPE"],
                "MAPE_global": metric["MAPE"],
                "RMSE_global": metric["RMSE"],
                "mape_zero_excluded_global": (
                    metric["mape_zero_excluded"]
                ),
                "n_observations_global": (
                    metric["n_observations"]
                ),
            })

        global_metrics = pd.DataFrame(
            global_rows
        )

        summary = macro.merge(
            global_metrics,
            on=["partition", "model"],
            how="left",
        )

        return (
            predictions,
            metrics_by_series,
            summary,
        )

    # ==============================================================
    # EJECUCIÓN COMPLETA
    # ==============================================================

    def run(self):

        self.load_data()

        partitions = self._get_partitions()

        prediction_frames = []
        series_metric_frames = []
        summary_frames = []

        for partition, train_dates, eval_dates in partitions:

            print(f"\nEvaluando {partition}:")

            print(
                f"  Entrenamiento: "
                f"{pd.to_datetime(train_dates).min().date()} "
                f"a {pd.to_datetime(train_dates).max().date()}"
            )

            print(
                f"  Evaluación: "
                f"{pd.to_datetime(eval_dates).min().date()} "
                f"a {pd.to_datetime(eval_dates).max().date()}"
            )

            if len(eval_dates) != self.horizon:
                raise ValueError(
                    f"{partition}: se esperaban "
                    f"{self.horizon} semanas y se recibieron "
                    f"{len(eval_dates)}."
                )

            preds, by_series, summary = (
                self.evaluate_partition(
                    partition,
                    train_dates,
                    eval_dates,
                )
            )

            prediction_frames.append(preds)
            series_metric_frames.append(by_series)
            summary_frames.append(summary)

        predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        metrics_by_series = pd.concat(
            series_metric_frames,
            ignore_index=True,
        )

        summary = pd.concat(
            summary_frames,
            ignore_index=True,
        )

        # Guardar predicciones.
        predictions.to_csv(
            self.output_dir
            / "predictions"
            / "all_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Guardar métricas por serie.
        metrics_by_series.to_csv(
            self.output_dir
            / "metrics"
            / "metrics_by_series.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Guardar resumen.
        summary.to_csv(
            self.output_dir
            / "metrics"
            / "validation_and_test_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        validation = summary[
            summary["partition"].str.startswith("fold_")
        ]

        test = summary[
            summary["partition"] == "final_test"
        ]

        validation.to_csv(
            self.output_dir
            / "metrics"
            / "validation_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        test.to_csv(
            self.output_dir
            / "metrics"
            / "test_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nRESUMEN DE MÉTRICAS — EXPERIMENTO B"
        )

        print(
            summary.to_string(index=False)
        )

        print(
            f"\nResultados guardados en: "
            f"{self.output_dir.resolve()}"
        )

        return summary


def main():
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
