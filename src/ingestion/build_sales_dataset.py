
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Cargar datos originales
    if settings["format"] == "csv":
        df = pd.read_csv(input_path)

    elif settings["format"] in ["xlsx", "xls"]:
        df = pd.read_excel(input_path)

    else:
        raise ValueError("Formato no soportado.")

    # 2. Normalizar encabezados
    df.columns = df.columns.astype(str).str.strip()

    date_col = settings["date_column"]
    quantity_col = settings["quantity_column"]
    price_col = settings["price_column"]

    category_col = settings["category_column"]
    store_col = settings["store_column"]
    gender_col = settings["gender_column"]

    # 3. Validar columnas
    required = [
        date_col,
        quantity_col,
        price_col,
        category_col,
        store_col,
        gender_col
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}"
        )

    # 4. Normalizar fechas
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        dayfirst=settings.get("dayfirst", False)
    )

    if df[date_col].isna().any():
        raise ValueError(
            "Existen fechas inválidas."
        )

    df["date"] = df[date_col].dt.normalize()

    # 5. Convertir variables numéricas
    df["quantity"] = pd.to_numeric(
        df[quantity_col],
        errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )

    if df[["quantity", "price"]].isna().any().any():
        raise ValueError(
            "Existen cantidades o precios inválidos."
        )

    if (df["quantity"] < 0).any():
        raise ValueError(
            "Existen cantidades negativas. "
            "Revisar devoluciones."
        )

    if (df["price"] < 0).any():
        raise ValueError(
            "Existen precios negativos."
        )

    # 6. Calcular importe de ventas
    df["sales"] = df["quantity"] * df["price"]

    # 7. Normalizar variables categóricas
    for column in [
        category_col,
        store_col,
        gender_col
    ]:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        if df[column].isna().any() or (df[column] == "").any():
            raise ValueError(
                f"Existen valores vacíos en {column}."
            )

    # 8. Renombrar dimensiones
    df.rename(
        columns={
            category_col: "category",
            store_col: "shopping_mall",
            gender_col: "gender"
        },
        inplace=True
    )

    # 9. Agregar por fecha, tienda, categoría y género
    sales_daily = (
        df.groupby(
            [
                "date",
                "shopping_mall",
                "category",
                "gender"
            ],
            as_index=False,
            observed=True
        )
        .agg(
            quantity=("quantity", "sum"),
            sales=("sales", "sum"),
            transactions=("quantity", "size")
        )
        .sort_values(
            [
                "date",
                "shopping_mall",
                "category",
                "gender"
            ]
        )
        .reset_index(drop=True)
    )

    # 10. Validar resultado
    if sales_daily.empty:
        raise ValueError(
            "No se generaron registros de ventas."
        )

    if sales_daily.duplicated(
        subset=[
            "date",
            "shopping_mall",
            "category",
            "gender"
        ]
    ).any():

        raise ValueError(
            "Existen combinaciones duplicadas."
        )

    # 11. Guardar dataset
    sales_daily.to_csv(
        output_path,
        index=False
    )

    # 12. Reporte
    print("\n======================================")
    print(" DATASET DE VENTAS DIMENSIONAL")
    print("======================================")

    print("Transacciones originales:", len(df))
    print("Registros agregados:", len(sales_daily))

    print("Fecha inicial:", df["date"].min())
    print("Fecha final:", df["date"].max())

    print("Demanda total:", sales_daily["quantity"].sum())
    print("Ventas totales:", sales_daily["sales"].sum())

    print(
        "Tiendas:",
        sales_daily["shopping_mall"].nunique()
    )

    print(
        "Categorías:",
        sales_daily["category"].nunique()
    )

    print(
        "Géneros:",
        sales_daily["gender"].nunique()
    )

    print("\nPrimeros registros:")
    print(sales_daily.head(10).to_string(index=False))

    print("\nArchivo generado:", output_path)

    print("\nPROCESO COMPLETADO")
    print("======================================")


if __name__ == "__main__":
    build_sales_dataset()
