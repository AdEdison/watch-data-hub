import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


OUTPUT_PATH = "data/watch_market_snapshot.csv"
LIGHT_OUTPUT_PATH = "data/watch_search_light.csv"
MANUAL_OVERRIDES_PATH = "data/manual_overrides.csv"

GITHUB_DATASET_API_URL = (
    "https://api.github.com/repos/philmorefkoung/"
    "Webscrapped-Watch-Dataset/contents/dataset?ref=main"
)

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

LIGHT_COLUMNS = [
    "display_name",
    "brand",
    "model",
    "reference_number",
    "reference_key",
    "search_text",
    "market_price_usd",
    "market_price_source",
    "market_price_updated_at",
    "shop_sources",
    "source_count",
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

    if text.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "-",
        "--",
    }:
        return ""

    return text


def normalize_reference(value) -> str:
    text = clean_text(value)

    if not text:
        return ""

    text = text.replace("Ref.", "")
    text = text.replace("REF.", "")
    text = text.replace("Reference", "")
    text = text.replace("reference", "")
    text = text.strip()

    # Разрешаем типичные символы часовых референсов:
    # буквы, цифры, точка, дефис, слэш.
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

    # Убираем валютные символы, пробелы и разделители тысяч.
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

    # Для Excel в русской локали безопаснее писать целые USD без ".0".
    return int(round(price))


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    exact = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in exact:
            return exact[key]

    # Мягкий поиск по вхождению.
    for col in df.columns:
        col_lower = str(col).strip().lower()
        for name in possible_names:
            if name.strip().lower() in col_lower:
                return col

    return None


def download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    # dtype=str критичен: референсы часов нельзя давать pandas/Excel
    # автоматически превращать в числа или даты.
    return pd.read_csv(StringIO(response.text), dtype=str)


def read_manual_overrides() -> pd.DataFrame:
    path = Path(MANUAL_OVERRIDES_PATH)

    if not path.exists():
        print(f"No manual overrides file found: {MANUAL_OVERRIDES_PATH}")
        return pd.DataFrame()

    overrides = pd.read_csv(path, dtype=str).fillna("")

    if overrides.empty:
        print("Manual overrides file is empty.")
        return pd.DataFrame()

    if "reference_key" not in overrides.columns:
        raise RuntimeError(
            "manual_overrides.csv must contain reference_key column."
        )

    overrides["reference_key"] = overrides["reference_key"].apply(make_reference_key)

    print(f"Loaded manual overrides: {len(overrides)} rows")

    return overrides

def apply_manual_overrides(snapshot: pd.DataFrame) -> pd.DataFrame:
    overrides = read_manual_overrides()

    if overrides.empty:
        return snapshot

    result = snapshot.copy()

    # Убираем дубли в overrides: последнее значение по reference_key считается актуальным.
    overrides = overrides.drop_duplicates(subset=["reference_key"], keep="last")

    result = result.merge(
        overrides,
        on="reference_key",
        how="left",
    )

    override_map = {
        "display_name_override": "display_name",
        "brand_override": "brand",
        "model_override": "model",
        "year_override": "year",
        "production_years_override": "production_years",
        "movement_override": "movement",
        "case_material_override": "case_material",
        "case_size_override": "case_size",
        "market_price_usd_override": "market_price_usd",
        "auction_price_usd_override": "auction_price_usd",
        "auction_house_override": "auction_house",
        "auction_date_override": "auction_date",
        "auction_lot_url_override": "auction_lot_url",
        "data_quality_override": "data_quality",
        "notes_override": "notes",
    }

    for override_col, target_col in override_map.items():
        if override_col not in result.columns:
            continue

        mask = result[override_col].notna() & (result[override_col].astype(str).str.strip() != "")

        if mask.any():
            result.loc[mask, target_col] = result.loc[mask, override_col]

    # Если руками изменили brand/model/display_name, пересобираем search_text.
    result["search_text"] = result.apply(
        lambda row: make_search_text(
            row.get("brand", ""),
            row.get("model", ""),
            row.get("reference_number", ""),
            row.get("reference_key", ""),
            row.get("display_name", ""),
            row.get("movement", ""),
            row.get("case_material", ""),
            row.get("case_size", ""),
            row.get("condition", ""),
            row.get("year", ""),
        ),
        axis=1,
    )

    # Пересобираем watch_id на случай изменения brand.
    result["watch_id"] = (
        result["brand"].astype(str)
        + "_"
        + result["reference_key"].astype(str)
    ).str.lower()

    result["watch_id"] = result["watch_id"].str.replace(
        r"[^a-z0-9]+",
        "_",
        regex=True,
    )
    result["watch_id"] = result["watch_id"].str.strip("_")

    # Убираем служебные override-колонки.
    override_cols = [col for col in result.columns if col.endswith("_override")]
    result = result.drop(columns=override_cols, errors="ignore")

    result = result[OUTPUT_COLUMNS].copy()

    print("Manual overrides applied.")

    return result

def guess_brand_from_filename(filename: str) -> str:
    name = filename.lower().replace(".csv", "").strip()
    compact = re.sub(r"[^a-z0-9]", "", name)

    mapping = {
        "rolex1": "Rolex",
        "rolex2": "Rolex",
        "rolex": "Rolex",
        "omega": "Omega",
        "omegaa": "Omega",
        "ap": "Audemars Piguet",
        "audemars": "Audemars Piguet",
        "patek": "Patek Philippe",
        "philippe": "Patek Philippe",
        "cartier": "Cartier",
        "tudor": "Tudor",
        "tagheuer": "TAG Heuer",
        "tag": "TAG Heuer",
        "iwc": "IWC",
        "longines": "Longines",
        "seiko": "Seiko",
        "breitling": "Breitling",
        "panerai": "Panerai",
        "hublot": "Hublot",
        "zenith": "Zenith",
        "jaegerlecoultre": "Jaeger-LeCoultre",
        "jaeger": "Jaeger-LeCoultre",
        "vacheron": "Vacheron Constantin",
        "constantin": "Vacheron Constantin",
        "richardmille": "Richard Mille",
        "richard": "Richard Mille",
        "mille": "Richard Mille",
        "breguet": "Breguet",
        "ulysse": "Ulysse Nardin",
        "nardin": "Ulysse Nardin",
        "hamilton": "Hamilton",
        "nomos": "NOMOS",
        "oris": "Oris",
        "sinn": "Sinn",
        "lange": "A. Lange & Söhne",
        "sohne": "A. Lange & Söhne",
        "a.lange": "A. Lange & Söhne",
    }

    # Сначала точные/более специфичные ключи.
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if key in compact:
            return mapping[key]

    # fallback: делаем имя файла читаемым.
    return (
        filename.replace(".csv", "")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def discover_source_files() -> list[dict]:
    print(f"Discovering source files from: {GITHUB_DATASET_API_URL}")

    response = requests.get(
        GITHUB_DATASET_API_URL,
        headers={"Accept": "application/vnd.github+json"},
        timeout=90,
    )
    response.raise_for_status()

    items = response.json()
    sources = []

    for item in items:
        if item.get("type") != "file":
            continue

        filename = item.get("name", "")

        if not filename.lower().endswith(".csv"):
            continue

        download_url = item.get("download_url")

        if not download_url:
            continue

        brand = guess_brand_from_filename(filename)

        sources.append(
            {
                "brand": brand,
                "url": download_url,
                "filename": filename,
            }
        )

    sources = sorted(sources, key=lambda x: (x["brand"], x["filename"]))

    print(f"Discovered {len(sources)} source CSV files:")

    for source in sources:
        print(f"- {source['brand']} | {source['filename']} | {source['url']}")

    return sources


def detect_columns(raw: pd.DataFrame) -> dict:
    columns = {
        "price": find_column(raw, ["price", "listing price", "usd price", "market price"]),
        "reference": find_column(
            raw,
            [
                "reference number",
                "reference_number",
                "reference",
                "ref",
                "reference no",
                "reference no.",
            ],
        ),
        "model": find_column(raw, ["model", "watch model", "name", "title"]),
        "movement": find_column(raw, ["movement", "caliber", "calibre"]),
        "condition": find_column(raw, ["condition", "watch condition"]),
        "year": find_column(raw, ["year", "production year", "manufacture year"]),
        "case_material": find_column(
            raw,
            ["case material", "case_material", "material", "case"],
        ),
        "case_size": find_column(
            raw,
            ["case size", "case_size", "diameter", "size"],
        ),
    }

    print("Detected columns:")
    for key, value in columns.items():
        print(f"- {key}: {value}")

    return columns


def build_snapshot() -> pd.DataFrame:
    date = today_utc()
    frames = []

    sources = discover_source_files()

    if not sources:
        raise RuntimeError("No source CSV files discovered.")

    for source in sources:
        brand = source["brand"]
        url = source["url"]
        filename = source.get("filename", "")

        print(f"Downloading {brand}: {filename} | {url}")

        try:
            df = download_csv(url)
        except Exception as exc:
            print(f"WARNING: failed to download {url}: {exc}")
            continue

        print(f"Downloaded rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        df["__brand_from_file"] = brand
        df["__source_url"] = url
        df["__source_filename"] = filename
        frames.append(df)

    if not frames:
        raise RuntimeError("No source data downloaded.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows: {len(raw)}")

    detected = detect_columns(raw)

    price_col = detected["price"]
    reference_col = detected["reference"]
    model_col = detected["model"]
    movement_col = detected["movement"]
    condition_col = detected["condition"]
    year_col = detected["year"]
    case_material_col = detected["case_material"]
    case_size_col = detected["case_size"]

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

        display_name = " ".join(
            part for part in [brand, model, reference] if clean_text(part)
        ).strip()

        search_text = make_search_text(
            brand,
            model,
            reference,
            reference_key,
            display_name,
            movement,
            case_material,
            case_size,
            condition,
            year,
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
        cleaned.groupby(
            ["brand", "model", "reference_number", "reference_key"],
            dropna=False,
        )
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
            source_urls=(
                "source_url",
                lambda x: " | ".join(sorted(set(map(str, x)))[:5]),
            ),
        )
        .reset_index()
    )

    grouped["market_price_usd"] = grouped["market_price_usd"].round(0).astype("Int64")

    grouped["watch_id"] = (
        grouped["brand"].astype(str)
        + "_"
        + grouped["reference_key"].astype(str)
    ).str.lower()

    grouped["watch_id"] = grouped["watch_id"].str.replace(
        r"[^a-z0-9]+",
        "_",
        regex=True,
    )
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

    grouped["notes"] = (
        "Median price from open historical/active listings dataset; "
        "not live appraisal."
    )
    grouped["updated_at"] = date

    result = grouped[OUTPUT_COLUMNS].copy()
    result = result.sort_values(["brand", "model", "reference_number"])

    print(f"Final grouped rows: {len(result)}")

    return result


def save_light_index(snapshot: pd.DataFrame) -> None:
    light = snapshot[LIGHT_COLUMNS].copy()

    # Сначала более надёжные записи, потом по бренду/названию.
    light["__quality_rank"] = (
        light["data_quality"]
        .map(
            {
                "high": 3,
                "medium": 2,
                "low": 1,
            }
        )
        .fillna(0)
    )

    light = light.sort_values(
        ["__quality_rank", "source_count", "brand", "display_name"],
        ascending=[False, False, True, True],
    )

    light = light.drop(columns=["__quality_rank"])

    light.to_csv(LIGHT_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(light)} rows to {LIGHT_OUTPUT_PATH}")


def ensure_data_dir() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)


def main():
    ensure_data_dir()

    snapshot = build_snapshot()
    snapshot = apply_manual_overrides(snapshot)

    snapshot.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(snapshot)} rows to {OUTPUT_PATH}")

    save_light_index(snapshot)

if __name__ == "__main__":
    main()
