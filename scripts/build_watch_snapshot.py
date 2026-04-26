import re
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests


OUTPUT_PATH = "data/watch_market_snapshot.csv"

SOURCE_FILES = [
    {
        "brand": "Rolex",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/rolex1.csv",
    },
    {
        "brand": "Rolex",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/rolex2.csv",
    },
    {
        "brand": "Omega",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/OMEGa.csv",
    },
    {
        "brand": "Audemars Piguet",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/AP.csv",
    },
    {
        "brand": "Patek Philippe",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/patek.csv",
    },
    {
        "brand": "Cartier",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/cartier.csv",
    },
    {
        "brand": "Tudor",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/tudor.csv",
    },
    {
        "brand": "TAG Heuer",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/tagheuer.csv",
    },
    {
        "brand": "Seiko",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/seiko.csv",
    },
    {
        "brand": "IWC",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/IWC.csv",
    },
    {
        "brand": "Longines",
        "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/Longines.csv",
    },
]


OUTPUT_COLUMNS = [
    "watch_id",
    "brand",
    "model",
    "reference_number",
    "display_name",
    "year",
    "production_years",
    "movement",
    "case_material",
    "case_size",
    "condition",
    "market_price_usd",
    "market_price_source",
    "market_price_updated_at",
    "auction_price_usd",
    "auction_house",
    "auction_date",
    "auction_lot_url",
    "shop_sources",
    "source_count",
    "source_urls",
    "data_quality",
    "notes",
    "updated_at",
]


def now_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def normalize_reference(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null", "unknown", "n/a", "-"}:
        return ""

    text = text.replace("Ref.", "")
    text = text.replace("Reference", "")
    text = text.strip()

    # Оставляем типичные символы референсов: буквы, цифры, точка, дефис, слэш.
    text = re.sub(r"[^A-Za-z0-9.\-/]", "", text)

    return text.upper()


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_price_to_float(value):
    if pd.isna(value):
        return None

    text = str(value)

    # Убираем валюты, пробелы и мусор.
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.]", "", text)

    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    # Мягкий поиск по вхождению.
    for col in df.columns:
        col_lower = str(col).strip().lower()
        for name in possible_names:
            if name.strip().lower() in col_lower:
                return col

    return None


def download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Некоторые CSV могут быть в разных кодировках. Сначала пробуем utf-8.
    text = response.text
    return pd.read_csv(StringIO(text))


def build_snapshot() -> pd.DataFrame:
    today = now_utc_date()
    raw_frames = []

    for source in SOURCE_FILES:
        brand = source["brand"]
        url = source["url"]

        print(f"Downloading {brand}: {url}")

        try:
            df = download_csv(url)
        except Exception as exc:
            print(f"ERROR: failed to download {url}: {exc}")
            continue

        df["__brand_from_file"] = brand
        df["__source_url"] = url
        raw_frames.append(df)

    if not raw_frames:
        raise RuntimeError("No source data downloaded.")

    raw = pd.concat(raw_frames, ignore_index=True)

    print("Raw columns:")
    print(list(raw.columns))

    # Подстраиваемся под разные названия колонок.
    price_col = find_column(raw, ["price", "Price", "listing price"])
    reference_col = find_column(raw, ["reference number", "reference_number", "reference", "ref"])
    model_col = find_column(raw, ["model", "Model", "watch model", "name", "title"])
    movement_col = find_column(raw, ["movement", "Movement"])
    condition_col = find_column(raw, ["condition", "Condition"])
    year_col = find_column(raw, ["year", "Year"])
    case_material_col = find_column(raw, ["case material", "case_material", "material"])
    case_size_col = find_column(raw, ["case size", "case_size", "diameter", "size"])

    if reference_col is None:
        raise RuntimeError(f"Reference column not found. Columns: {list(raw.columns)}")

    if price_col is None:
        raise RuntimeError(f"Price column not found. Columns: {list(raw.columns)}")

    rows = []

    for _, item in raw.iterrows():
        brand = normalize_text(item.get("__brand_from_file", ""))
        reference = normalize_reference(item.get(reference_col, ""))

        if reference == "":
            continue

        model = normalize_text(item.get(model_col, "")) if model_col else ""
        movement = normalize_text(item.get(movement_col, "")) if movement_col else ""
        condition = normalize_text(item.get(condition_col, "")) if condition_col else ""
        year = normalize_text(item.get(year_col, "")) if year_col else ""
        case_material = normalize_text(item.get(case_material_col, "")) if case_material_col else ""
        case_size = normalize_text(item.get(case_size_col, "")) if case_size_col else ""
        price = parse_price_to_float(item.get(price_col, ""))

        if price is None or price <= 0:
            continue

        display_name = " ".join(part for part in [brand, model, reference] if part).strip()

        rows.append(
            {
                "brand": brand,
                "model": model,
                "reference_number": reference,
                "display_name": display_name,
                "year": year,
                "production_years": "",
                "movement": movement,
                "case_material": case_material,
                "case_size": case_size,
                "condition": condition,
                "price": price,
                "source_url": item.get("__source_url", ""),
            }
        )

    cleaned = pd.DataFrame(rows)

    if cleaned.empty:
        raise RuntimeError("Cleaned dataset is empty. Check column mapping.")

    # Группируем по brand + model + reference.
    grouped = (
        cleaned.groupby(["brand", "model", "reference_number"], dropna=False)
        .agg(
            display_name=("display_name", "first"),
            year=("year", "first"),
            production_years=("production_years", "first"),
            movement=("movement", "first"),
            case_material=("case_material", "first"),
            case_size=("case_size", "first"),
            condition=("condition", "first"),
            market_price_usd=("price", "median"),
            source_count=("price", "count"),
            source_urls=("source_url", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
        )
        .reset_index()
    )

    grouped["watch_id"] = (
        grouped["brand"].astype(str)
        + "_"
        + grouped["reference_number"].astype(str)
    ).str.lower()

    grouped["watch_id"] = grouped["watch_id"].str.replace(r"[^a-z0-9]+", "_", regex=True)
    grouped["watch_id"] = grouped["watch_id"].str.strip("_")

    grouped["market_price_usd"] = grouped["market_price_usd"].round(2)
    grouped["market_price_source"] = "webscrapped_watch_dataset_median"
    grouped["market_price_updated_at"] = today

    grouped["auction_price_usd"] = ""
    grouped["auction_house"] = ""
    grouped["auction_date"] = ""
    grouped["auction_lot_url"] = ""

    grouped["shop_sources"] = "Chrono24 dataset"
    grouped["data_quality"] = grouped["source_count"].apply(
        lambda count: "medium" if count >= 3 else "low"
    )
    grouped["notes"] = "Median price from open dataset; not live market price."
    grouped["updated_at"] = today

    result = grouped[OUTPUT_COLUMNS].copy()

    # Сортировка для удобства.
    result = result.sort_values(["brand", "model", "reference_number"])

    return result


def main():
    snapshot = build_snapshot()
    snapshot.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(snapshot)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
