
import pandas as pd
import yaml
import requests

from pathlib import Path
from io import BytesIO


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_initial():

    config = load_config()
    settings = config["data"]["initial"]

    output_dir = Path(config["data"]["raw_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    source = settings["source"]
    file_path = settings["file_path"]
    file_format = settings["format"].lower()

    # 1. Leer dataset
    if source == "local":

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No se encuentra el dataset original: {path}"
            )

        if file_format == "csv":
            df = pd.read_csv(path)
        elif file_format in ["xlsx", "xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError("Formato no soportado.")

    elif source == "url":

        url = settings["url"]

        if not url:
            raise ValueError("No se configuró la URL.")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        if file_format == "csv":
            df = pd.read_csv(BytesIO(response.content))
        elif file_format in ["xlsx", "xls"]:
            df = pd.read_excel(BytesIO(response.content))
        else:
            raise ValueError("Formato no soportado.")

    else:
        raise ValueError("Source debe ser local o url.")

    # 2. Validar columnas
    date_col = settings["date_column"]
    target_col = settings["target_column"]

    required = [date_col, target_col]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing}"
        )

    # 3. Normalizar nombres de columnas
    df.columns = df.columns.str.strip()

    date_col = date_col.strip()
    target_col = target_col.strip()

    # 4. Normalizar fechas
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    if df[date_col].isna().any():
        raise ValueError(
            "Existen fechas inválidas en el dataset original."
        )

    # 5. Ordenar
    df = df.sort_values(date_col).reset_index(drop=True)

    # 6. Guardar
    output_path = output_dir / "initial_dataset.csv"

    df.to_csv(output_path, index=False)

    # 7. Reporte
    print("\nDATASET ORIGINAL DESCARGADO")
    print("---------------------------")
    print("Registros:", len(df))
    print("Columnas:", list(df.columns))
    print("Fecha inicial:", df[date_col].min())
    print("Fecha final:", df[date_col].max())
    print("Variable objetivo:", target_col)
    print("Archivo:", output_path)


if __name__ == "__main__":
    download_initial()
