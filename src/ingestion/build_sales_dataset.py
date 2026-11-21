
import pandas as pd
import yaml

from pathlib import Path


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_sales_dataset():

    config = load_config()
    settings = config["sales"]

    input_path = Path(settings["input_file"])
    output_path = Path(settings["output_file"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos originales
    if settings["format"] == "csv":
        df = pd.read_csv(input_path)
    elif settings["format"] in ["xlsx", "xls"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Formato no soportado")

    # 2. Normalizar encabezados
    df.columns = df.columns.astype(str).str.strip()

    date_col = settings["date_column"]
    mode = settings["sales_mode"]

    # 3. Validar columnas
    required = [date_col]

    if mode == "amount":
        required.append(settings["amount_column"])

    elif mode == "quantity_price":
        required.extend([
            settings["quantity_column"],
            settings["price_column"]
        ])

    else:
        raise ValueError(
            "sales_mode debe ser amount o quantity_price"
        )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}. "
            f"Disponibles: {df.columns.tolist()}"
        )

    # 4. Convertir fechas
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        dayfirst=settings.get("dayfirst", False)
    )

    if df[date_col].isna().any():
        raise ValueError(
            "Existen fechas inválidas. Revisar los datos originales."
        )

    # 5. Calcular ventas
    if mode == "amount":

        df["sales"] = pd.to_numeric(
            df[settings["amount_column"]],
            errors="coerce"
        )

    else:

        quantity = pd.to_numeric(
            df[settings["quantity_column"]],
            errors="coerce"
        )

        price = pd.to_numeric(
            df[settings["price_column"]],
            errors="coerce"
        )

        df["sales"] = quantity * price

    if df["sales"].isna().any():
        raise ValueError(
            "Existen importes o cantidades inválidas."
        )

    # 6. Agregar ventas por día
    df["date"] = df[date_col].dt.normalize()

    sales_daily = (
        df.groupby("date", as_index=False)
        .agg(sales=("sales", "sum"))
        .sort_values("date")
    )

    # 7. Validar resultado
    if sales_daily.empty:
        raise ValueError("No se generaron ventas.")

    if (sales_daily["sales"] < 0).any():
        print(
            "AVISO: existen ventas netas negativas. "
            "Revisar devoluciones y cancelaciones."
        )

    # 8. Guardar dataset
    sales_daily.to_csv(output_path, index=False)

    # 9. Reporte
    print("\n========== DATASET DE VENTAS ==========")
    print("Registros originales:", len(df))
    print("Registros diarios:", len(sales_daily))
    print("Fecha inicial:", sales_daily["date"].min())
    print("Fecha final:", sales_daily["date"].max())
    print("Ventas totales:", sales_daily["sales"].sum())
    print("Promedio diario:", sales_daily["sales"].mean())
    print("Archivo generado:", output_path)

    print("\nPrimeros registros:")
    print(sales_daily.head(10))

    print("\n=======================================")


if __name__ == "__main__":
    build_sales_dataset()
