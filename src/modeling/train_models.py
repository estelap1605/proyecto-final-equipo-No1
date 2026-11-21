"""
Entrenamiento y evaluación de Regresión Lineal, SARIMAX y LSTM.
Experimento B exclusivamente: demanda histórica + calendario + clima.

Ejecución desde la raíz del proyecto:
    python -m src.modeling.train_models

Entradas:
    data/processed/model_preparation/series_dataset.csv
    data/processed/model_preparation/supervised_dataset.csv
    data/processed/model_preparation/feature_selection.json
    data/processed/model_preparation/split_metadata.json

Salidas:
    data/processed/model_results/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# TensorFlow se importa de forma diferida para que los otros modelos
# puedan ejecutarse y diagnosticarse aunque TensorFlow no esté instalado.


class ModelTrainer:
    """Entrena y evalúa los tres modelos con pronósticos de origen fijo."""

    def __init__(
        self,
        preparation_dir: str | Path = "data/processed/model_preparation",
        output_dir: str | Path = "data/processed/model_results",
        horizon: int = 4,
        lstm_lookback: int = 8,
        random_state: int = 42,
        lstm_epochs: int = 60,
        lstm_batch_size: int = 16,
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # CARGA
    # ------------------------------------------------------------------

    def load_data(self):
        required = [
            self.series_path,
            self.supervised_path,
            self.selection_path,
            self.splits_path,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "No se encontraron los siguientes archivos:\n" + "\n".join(missing)
            )

        self.series = pd.read_csv(self.series_path, parse_dates=["week_start"])
        self.supervised = pd.read_csv(
            self.supervised_path, parse_dates=["week_start"]
        )
        self.feature_selection = json.loads(
            self.selection_path.read_text(encoding="utf-8")
        )
        self.split_metadata = json.loads(
            self.splits_path.read_text(encoding="utf-8")
        )

        keys = {"week_start", "shopping_mall", "category", "quantity"}
        for label, frame in [
            ("series_dataset.csv", self.series),
            ("supervised_dataset.csv", self.supervised),
        ]:
            absent = keys.difference(frame.columns)
            if absent:
                raise ValueError(f"{label}: faltan columnas {sorted(absent)}")

        self.series = self.series.sort_values(
            ["shopping_mall", "category", "week_start"]
        ).reset_index(drop=True)
        self.supervised = self.supervised.sort_values(
            ["shopping_mall", "category", "week_start"]
        ).reset_index(drop=True)

        count = self.series[["shopping_mall", "category"]].drop_duplicates().shape[0]
        if count != 80:
            raise ValueError(f"Se esperaban 80 series; se encontraron {count}.")

        print(f"Series cargadas: {count}")
        print(f"Registros históricos: {len(self.series):,}")
        print(f"Registros supervisados: {len(self.supervised):,}")
        return self

    # ------------------------------------------------------------------
    # PARTICIONES Y SELECCIÓN DE VARIABLES
    # ------------------------------------------------------------------

    @staticmethod
    def _date_list(values):
        return pd.to_datetime(values).tolist()

    def _get_partitions(self):
        """Lee folds/final_test del JSON, admitiendo las estructuras generadas."""
        meta = self.split_metadata
        partitions = []

        folds = meta.get("folds", [])
        if isinstance(folds, dict):
            folds = list(folds.values())

        for i, fold in enumerate(folds, start=1):
            train_dates = fold.get("train_dates", fold.get("train", []))
            val_dates = fold.get(
                "validation_dates", fold.get("val_dates", fold.get("validation", []))
            )
            if train_dates and val_dates:
                partitions.append(
                    (f"fold_{fold.get('fold', i)}",
                     self._date_list(train_dates),
                     self._date_list(val_dates))
                )

        final = meta.get("final_test", {})
        if final:
            train_dates = final.get(
                "train_dates", meta.get("development_dates", [])
            )
            test_dates = final.get("test_dates", meta.get("test_dates", []))
        else:
            train_dates = meta.get("development_dates", [])
            test_dates = meta.get("test_dates", [])

        if train_dates and test_dates:
            partitions.append(
                ("final_test", self._date_list(train_dates), self._date_list(test_dates))
            )

        if not partitions:
            raise ValueError(
                "No pude interpretar folds/final_test en split_metadata.json. "
                "Se esperaba train_dates y validation_dates/test_dates."
            )
        return partitions

    @staticmethod
    def _series_key(mall, category):
        return f"{mall} | {category}"

    def _selected_features(self, partition, model_name, mall, category, candidates):
        """
        Recupera la selección específica de modelo/partición/serie.
        La estructura esperada es partition -> model -> 'mall | category' -> lista.
        """
        node = self.feature_selection
        for key in (partition, model_name, self._series_key(mall, category)):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break

        if isinstance(node, list):
            selected = node
        elif isinstance(node, dict):
            selected = node.get("selected_features", node.get("features", []))
        else:
            selected = []

        selected = [c for c in selected if c in candidates]
        if not selected:
            # Evita detener todo el experimento si una serie/fold no tuvo
            # variables seleccionables (caso posible en series constantes).
            selected = list(candidates)
        return selected

    @staticmethod
    def _feature_columns(frame):
        excluded = {"week_start", "shopping_mall", "category", "quantity"}
        return [
            c for c in frame.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
        ]

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[valid], y_pred[valid]

        if len(y_true) == 0:
            return {"WAPE": np.nan, "MAPE": np.nan, "RMSE": np.nan,
                    "mape_zero_excluded": 0, "n_observations": 0}

        denominator = np.abs(y_true).sum()
        wape = (
            100.0 * np.abs(y_true - y_pred).sum() / denominator
            if denominator > 0 else np.nan
        )

        nonzero = y_true != 0
        mape = (
            100.0 * np.mean(
                np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
            )
            if nonzero.any() else np.nan
        )
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        return {
            "WAPE": float(wape) if np.isfinite(wape) else np.nan,
            "MAPE": float(mape) if np.isfinite(mape) else np.nan,
            "RMSE": rmse,
            "mape_zero_excluded": int((~nonzero).sum()),
            "n_observations": int(len(y_true)),
        }

    # ------------------------------------------------------------------
    # CONSTRUCCIÓN DE FILAS PARA PRONÓSTICO RECURSIVO
    # ------------------------------------------------------------------

    def _make_future_row(self, history, template, feature_columns):
        """Actualiza rezagos/rolling con demanda observada + predicciones."""
        row = template.copy()
        values = np.asarray(history, dtype=float)

        for col in feature_columns:
            if col.startswith("quantity_lag_"):
                lag = int(col.rsplit("_", 1)[1])
                row[col] = values[-lag] if len(values) >= lag else np.nan

            elif col.startswith("quantity_rolling_mean_"):
                window = int(col.rsplit("_", 1)[1])
                row[col] = np.mean(values[-window:]) if len(values) >= window else np.nan

            elif col.startswith("quantity_rolling_std_"):
                window = int(col.rsplit("_", 1)[1])
                row[col] = np.std(values[-window:], ddof=0) if len(values) >= window else np.nan

        return row

    @staticmethod
    def _clean_matrix(train_x, future_x):
        train_x = train_x.replace([np.inf, -np.inf], np.nan)
        future_x = future_x.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        train_array = imputer.fit_transform(train_x)
        future_array = imputer.transform(future_x)
        return train_array, future_array, imputer

    # ------------------------------------------------------------------
    # REGRESIÓN LINEAL
    # ------------------------------------------------------------------

    def train_linear_regression(
        self, train_frame, future_templates, features
    ):
        X_train = train_frame[features].copy()
        y_train = train_frame["quantity"].astype(float).to_numpy()
        X_future = future_templates[features].copy()
        X_train, X_future, imputer = self._clean_matrix(X_train, X_future)

        model = LinearRegression()
        model.fit(X_train, y_train)

        history = train_frame["quantity"].astype(float).tolist()
        predictions = []
        for step in range(len(future_templates)):
            row = future_templates.iloc[[step]].copy()
            dynamic = self._make_future_row(history, row.iloc[0], features)
            x = pd.DataFrame([dynamic])[features].replace(
                [np.inf, -np.inf], np.nan
            )
            x = imputer.transform(x)
            prediction = max(0.0, float(model.predict(x)[0]))
            predictions.append(prediction)
            history.append(prediction)

        return predictions, model

    # ------------------------------------------------------------------
    # SARIMAX
    # ------------------------------------------------------------------

    def train_sarimax(self, train_frame, future_templates, features):
        y_train = train_frame["quantity"].astype(float).to_numpy()
        X_train = train_frame[features].copy()
        X_future = future_templates[features].copy()
        X_train, X_future, _ = self._clean_matrix(X_train, X_future)

        # Modelo parsimonioso: el tamaño de cada serie es reducido.
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.get_forecast(
            steps=len(future_templates), exog=X_future
        ).predicted_mean
        predictions = np.maximum(0.0, np.asarray(forecast, dtype=float))
        return predictions.tolist(), fitted

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------

    def train_lstm(self, train_frame, future_templates, features):
        try:
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.optimizers import Adam
        except ImportError as exc:
            raise ImportError(
                "TensorFlow no está instalado en el entorno activo. "
                "Instálalo en el entorno Python 3.12."
            ) from exc

        np.random.seed(self.random_state)
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        X_raw = train_frame[features].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_raw)
        y = train_frame["quantity"].astype(float).to_numpy().reshape(-1, 1)

        x_scaler = StandardScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y).ravel()

        lookback = self.lstm_lookback
        if len(X_scaled) <= lookback + 3:
            lookback = max(2, min(lookback, len(X_scaled) - 4))

        sequences, targets = [], []
        for i in range(lookback, len(X_scaled)):
            sequences.append(X_scaled[i - lookback:i])
            targets.append(y_scaled[i])

        if not sequences:
            raise ValueError("No hay suficientes observaciones para crear secuencias LSTM.")

        X_seq = np.asarray(sequences, dtype=np.float32)
        y_seq = np.asarray(targets, dtype=np.float32)

        model = Sequential([
            LSTM(32, input_shape=(lookback, len(features)), return_sequences=False),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        early_stop = EarlyStopping(
            monitor="loss", patience=8, restore_best_weights=True
        )
        model.fit(
            X_seq, y_seq,
            epochs=self.lstm_epochs,
            batch_size=min(self.lstm_batch_size, len(X_seq)),
            verbose=0,
            callbacks=[early_stop],
            shuffle=False,
        )

        # Pronóstico recursivo: las variables de demanda de cada nuevo
        # paso se calculan con valores reales previos + predicciones.
        history = train_frame["quantity"].astype(float).tolist()
        feature_history = [row.copy() for row in X_scaled]
        predictions = []

        for step in range(len(future_templates)):
            template = future_templates.iloc[step].copy()
            dynamic = self._make_future_row(history, template, features)
            row_df = pd.DataFrame([dynamic])[features].replace(
                [np.inf, -np.inf], np.nan
            )
            row_scaled = x_scaler.transform(imputer.transform(row_df))[0]
            feature_history.append(row_scaled)

            sequence = np.asarray(
                feature_history[-lookback:], dtype=np.float32
            )
            if len(sequence) < lookback:
                pad = np.repeat(sequence[:1], lookback - len(sequence), axis=0)
                sequence = np.vstack([pad, sequence])

            pred_scaled = float(model.predict(sequence[None, :, :], verbose=0)[0, 0])
            prediction = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
            prediction = max(0.0, prediction)
            predictions.append(prediction)
            history.append(prediction)

        return predictions, model

    # ------------------------------------------------------------------
    # EVALUACIÓN DE UNA PARTICIÓN
    # ------------------------------------------------------------------

    def evaluate_partition(self, partition, train_dates, eval_dates):
        all_predictions = []
        series_metrics = []

        models = {
            "linear_regression": self.train_linear_regression,
            "sarimax": self.train_sarimax,
            "lstm": self.train_lstm,
        }

        train_dates = pd.to_datetime(train_dates)
        eval_dates = pd.to_datetime(eval_dates)

        for (mall, category), full_series in self.series.groupby(
            ["shopping_mall", "category"], sort=True
        ):
            full_series = full_series.sort_values("week_start").reset_index(drop=True)
            sup = self.supervised[
                (self.supervised["shopping_mall"] == mall)
                & (self.supervised["category"] == category)
            ].sort_values("week_start").reset_index(drop=True)

            train = sup[sup["week_start"].isin(train_dates)].copy()
            future = sup[sup["week_start"].isin(eval_dates)].copy()
            future = future.sort_values("week_start").reset_index(drop=True)

            if len(future) != len(eval_dates):
                raise ValueError(
                    f"{mall} | {category}: faltan filas futuras en {partition}."
                )
            if train.empty:
                raise ValueError(f"{mall} | {category}: entrenamiento vacío.")

            # Asegura que el entrenamiento termine antes del primer objetivo.
            if train["week_start"].max() >= future["week_start"].min():
                raise ValueError("Se detectó solapamiento temporal train/evaluación.")

            candidates = self._feature_columns(train)
            for model_name, trainer in models.items():
                features = self._selected_features(
                    partition, model_name, mall, category, candidates
                )

                try:
                    preds, fitted = trainer(train, future, features)
                    status = "ok"
                    error = ""
                    try:
                        model_path = (
                            self.output_dir / "models" / partition
                            / model_name / f"{self._safe_name(mall)}__{self._safe_name(category)}.pkl"
                        )
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        import joblib
                        # Los modelos Keras se guardan en formato .keras.
                        if model_name == "lstm":
                            fitted.save(model_path.with_suffix(".keras"))
                        else:
                            joblib.dump(fitted, model_path)
                    except Exception as save_exc:
                        warnings.warn(f"No se pudo guardar modelo: {save_exc}")
                except Exception as exc:
                    preds = [np.nan] * len(future)
                    status = "error"
                    error = str(exc)
                    warnings.warn(
                        f"Error {partition} / {model_name} / {mall} / {category}: {exc}"
                    )

                for date, actual, pred in zip(
                    future["week_start"], future["quantity"], preds
                ):
                    all_predictions.append({
                        "partition": partition,
                        "model": model_name,
                        "shopping_mall": mall,
                        "category": category,
                        "week_start": date,
                        "y_true": float(actual),
                        "y_pred": float(pred) if np.isfinite(pred) else np.nan,
                        "status": status,
                        "error": error,
                    })

                valid_preds = np.isfinite(np.asarray(preds, dtype=float))
                if valid_preds.any():
                    metrics = self.calculate_metrics(
                        future.loc[valid_preds, "quantity"].to_numpy(),
                        np.asarray(preds, dtype=float)[valid_preds],
                    )
                else:
                    metrics = self.calculate_metrics([], [])

                series_metrics.append({
                    "partition": partition,
                    "model": model_name,
                    "shopping_mall": mall,
                    "category": category,
                    **metrics,
                    "status": status,
                    "error": error,
                })

        predictions = pd.DataFrame(all_predictions)
        metrics_by_series = pd.DataFrame(series_metrics)

        # Resumen macro: promedio de las métricas de las 80 series.
        macro = (
            metrics_by_series.groupby(["partition", "model"], dropna=False)
            .agg(
                WAPE_mean_series=("WAPE", "mean"),
                MAPE_mean_series=("MAPE", "mean"),
                RMSE_mean_series=("RMSE", "mean"),
                successful_series=("status", lambda x: int((x == "ok").sum())),
                total_series=("status", "size"),
            )
            .reset_index()
        )

        # Métricas globales: concatenación de todas las observaciones.
        global_rows = []
        for (part, model_name), group in predictions.groupby(["partition", "model"]):
            valid = group[["y_true", "y_pred"]].notna().all(axis=1)
            metric = self.calculate_metrics(
                group.loc[valid, "y_true"], group.loc[valid, "y_pred"]
            )
            global_rows.append({
                "partition": part,
                "model": model_name,
                "WAPE_global": metric["WAPE"],
                "MAPE_global": metric["MAPE"],
                "RMSE_global": metric["RMSE"],
                "mape_zero_excluded_global": metric["mape_zero_excluded"],
                "n_observations_global": metric["n_observations"],
            })

        global_metrics = pd.DataFrame(global_rows)
        summary = macro.merge(global_metrics, on=["partition", "model"], how="left")

        return predictions, metrics_by_series, summary

    # ------------------------------------------------------------------
    # EJECUCIÓN COMPLETA
    # ------------------------------------------------------------------

    def run(self):
        self.load_data()
        partitions = self._get_partitions()

        prediction_frames = []
        series_metric_frames = []
        summary_frames = []

        for partition, train_dates, eval_dates in partitions:
            print(f"\nEvaluando {partition}:")
            print(
                f"  Entrenamiento: {pd.to_datetime(train_dates).min().date()} "
                f"a {pd.to_datetime(train_dates).max().date()}"
            )
            print(
                f"  Evaluación: {pd.to_datetime(eval_dates).min().date()} "
                f"a {pd.to_datetime(eval_dates).max().date()}"
            )
            if len(eval_dates) != self.horizon:
                raise ValueError(
                    f"{partition}: se esperaban {self.horizon} semanas "
                    f"y se recibieron {len(eval_dates)}."
                )

            preds, by_series, summary = self.evaluate_partition(
                partition, train_dates, eval_dates
            )
            prediction_frames.append(preds)
            series_metric_frames.append(by_series)
            summary_frames.append(summary)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        metrics_by_series = pd.concat(series_metric_frames, ignore_index=True)
        summary = pd.concat(summary_frames, ignore_index=True)

        predictions.to_csv(
            self.output_dir / "predictions" / "all_predictions.csv",
            index=False, encoding="utf-8-sig"
        )
        metrics_by_series.to_csv(
            self.output_dir / "metrics" / "metrics_by_series.csv",
            index=False, encoding="utf-8-sig"
        )
        summary.to_csv(
            self.output_dir / "metrics" / "validation_and_test_metrics.csv",
            index=False, encoding="utf-8-sig"
        )

        validation = summary[summary["partition"].str.startswith("fold_")]
        test = summary[summary["partition"] == "final_test"]

        validation.to_csv(
            self.output_dir / "metrics" / "validation_metrics.csv",
            index=False, encoding="utf-8-sig"
        )
        test.to_csv(
            self.output_dir / "metrics" / "test_metrics.csv",
            index=False, encoding="utf-8-sig"
        )

        print("\nRESUMEN DE MÉTRICAS — EXPERIMENTO B")
        print(summary.to_string(index=False))
        print(f"\nResultados guardados en: {self.output_dir.resolve()}")
        return summary


def main():
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
