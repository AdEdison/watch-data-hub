import re
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests


OUTPUT_PATH = "data/watch_market_snapshot.csv"

SOURCE_FILES = [
    {"brand": "Rolex", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/rolex1.csv"},
    {"brand": "Rolex", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/rolex2.csv"},
    {"brand": "Omega", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/OMEGa.csv"},
    {"brand": "Audemars Piguet", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/AP.csv"},
    {"brand": "Patek Philippe", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/patek.csv"},
    {"brand": "Cartier", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/cartier.csv"},
    {"brand": "Tudor", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/tudor.csv"},
    {"brand": "TAG Heuer", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/tagheuer.csv"},
    {"brand": "Seiko", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/seiko.csv"},
    {"brand": "IWC", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/IWC.csv"},
    {"brand": "Longines", "url": "https://raw.githubusercontent.com/philmorefkoung/Webscrapped-Watch-Dataset/main/dataset/Longines.csv"},
]

OUTPUT_COLUMNS = [
    "watch_id",
    "brand",
    "model",
    "reference_number",
    "reference_key",
    "display_name",
    "search_text",
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


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "n/a", "unknown", "-"}:
        return ""
    return text


def normalize_reference(value) -> str:
    text = clean_text(value)
    if not text:
        return ""

    text = text.replace("Ref.", "")
    text = text.replace("REF.", "")
    text = text.replace("Reference", "")
    text = text.strip()

    # Разрешаем типичные символы референсов часов.
    text = re.sub(r"[^A-Za-z0-9.\-/]", "", text)

    return text.upper()


def make_reference_key(reference: str) -> str:
    """
    15091BC.OO.D002CR.01 -> 15091BCOOD002CR01
    15091BC/OO/D002CR/01 -> 15091BCOOD002CR01
    """
    return re.sub(r"[^A-Z0-9]", "", normalize_reference(reference))


def make_search_text(*parts) -> str:
    joined = " ".join(clean_text(p) for p in parts if clean_text(p))
    joined = joined.upper()
    joined = re.sub(r"[^A-Z0-9А-ЯЁ]+", " ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def parse_price(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    # Убираем валютные символы и пробелы.
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.]", "", text)

    if not text:
        return None

    try:
        price = float(text)
    except ValueError:
        return None

    if price <= 0:
        return None

    # Для Excel в русской локали безопаснее отдавать целые USD без десятичной точки.
    return int(round(price))


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    exact = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in exact:
            return exact[key]

    for col in df.columns:
        col_lower = str(col).strip().lower()
        for name in possible_names:
            if name.strip().lower() in col_lower:
                return col

    return None


def download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), dtype=str)


def build_snapshot() -> pd.DataFrame:
    date = today_utc()
    frames = []

    for source in SOURCE_FILES:
        brand = source["brand"]
        url = source["url"]

        print(f"Downloading {brand}: {url}")

        try:
            df = download_csv(url)
        except Exception as exc:
            print(f"WARNING: failed to download {url}: {exc}")
            continue

        print(f"Downloaded rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        df["__brand_from_file"] = brand
        df["__source_url"] = url
        frames.append(df)

    if not frames:
        raise RuntimeError("No source data downloaded.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows: {len(raw)}")

    price_col = find_column(raw, ["price", "listing price", "usd price"])
    reference_col = find_column(raw, ["reference number", "reference_number", "reference", "ref"])
    model_col = find_column(raw, ["model", "watch model", "name", "title"])
    movement_col = find_column(raw, ["movement"])
    condition_col = find_column(raw, ["condition"])
    year_col = find_column(raw, ["year"])
    case_material_col = find_column(raw, ["case material", "case_material", "material"])
    case_size_col = find_column(raw, ["case size", "case_size", "diameter", "size"])

    print(f"Detected price_col: {price_col}")
    print(f"Detected reference_col: {reference_col}")
    print(f"Detected model_col: {model_col}")
    print(f"Detected movement_col: {movement_col}")

    if reference_col is None:
        raise RuntimeError(f"Reference column not found. Columns: {list(raw.columns)}")

    if price_col is None:
        raise RuntimeError(f"Price column not found. Columns: {list(raw.columns)}")

    rows = []

    for _, item in raw.iterrows():
        brand = clean_text(item.get("__brand_from_file", ""))
        reference = normalize_reference(item.get(reference_col, ""))

        if not reference:
            continue

        reference_key = make_reference_key(reference)

        if not reference_key:
            continue

        price = parse_price(item.get(price_col, ""))

        if price is None:
            continue

        model = clean_text(item.get(model_col, "")) if model_col else ""
        movement = clean_text(item.get(movement_col, "")) if movement_col else ""
        condition = clean_text(item.get(condition_col, "")) if condition_col else ""
        year = clean_text(item.get(year_col, "")) if year_col else ""
        case_material = clean_text(item.get(case_material_col, "")) if case_material_col else ""
        case_size = clean_text(item.get(case_size_col, "")) if case_size_col else ""

        display_name = " ".join(part for part in [brand, model, reference] if part).strip()

        search_text = make_search_text(
            brand,
            model,
            reference,
            reference_key,
            display_name,
            movement,
            case_material,
            case_size,
        )

        rows.append(
            {
                "brand": brand,
                "model": model,
                "reference_number": reference,
                "reference_key": reference_key,
                "display_name": display_name,
                "search_text": search_text,
                "year": year,
                "production_years": "",
                "movement": movement,
                "case_material": case_material,
                "case_size": case_size,
                "condition": condition,
                "price": price,
                "source_url": clean_text(item.get("__source_url", "")),
            }
        )

    cleaned = pd.DataFrame(rows)

    if cleaned.empty:
        raise RuntimeError("Cleaned dataset is empty. Check source columns and parsing.")

    print(f"Cleaned rows with price and reference: {len(cleaned)}")

    grouped = (
        cleaned.groupby(["brand", "model", "reference_number", "reference_key"], dropna=False)
        .agg(
            display_name=("display_name", "first"),
            search_text=("search_text", "first"),
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

    grouped["market_price_usd"] = grouped["market_price_usd"].round(0).astype("Int64")

    grouped["watch_id"] = (
        grouped["brand"].astype(str)
        + "_"
        + grouped["reference_key"].astype(str)
    ).str.lower()

    grouped["watch_id"] = grouped["watch_id"].str.replace(r"[^a-z0-9]+", "_", regex=True)
    grouped["watch_id"] = grouped["watch_id"].str.strip("_")

    grouped["market_price_source"] = "webscrapped_watch_dataset_median"
    grouped["market_price_updated_at"] = date

    grouped["auction_price_usd"] = ""
    grouped["auction_house"] = ""
    grouped["auction_date"] = ""
    grouped["auction_lot_url"] = ""

    grouped["shop_sources"] = "Chrono24 dataset"
    grouped["data_quality"] = grouped["source_count"].apply(
        lambda count: "medium" if int(count) >= 3 else "low"
    )
    grouped["notes"] = "Median price from open historical/active listings dataset; not live appraisal."
    grouped["updated_at"] = date

    result = grouped[OUTPUT_COLUMNS].copy()
    result = result.sort_values(["brand", "model", "reference_number"])

    return result


def main():
    snapshot = build_snapshot()
    snapshot.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(snapshot)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
