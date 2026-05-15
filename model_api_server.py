import json
import pickle
import hashlib
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_model"
DATA_DIR = ROOT / "data"


def _canon(text):
    return " ".join(str(text or "").split()).strip()


def _season_from_month(month):
    return "Kharif" if month in {6, 7, 8, 9, 10, 11} else "Rabi"


def _find_col(columns, candidates):
    lowered = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for col in columns:
        key = str(col).lower().strip()
        if any(cand in key for cand in candidates):
            return col
    return None


def _load_artifacts():
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(MODEL_DIR / "feature_list.pkl", "rb") as f:
        feature_list = pickle.load(f)
    return model, scaler, label_encoders, feature_list


def _load_data():
    paths = [
        DATA_DIR / "mandi_prices_dataset1.csv",
        DATA_DIR / "mandi_prices_dataset2.csv",
    ]
    frames = [pd.read_csv(p, low_memory=False) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame(), {}

    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns)
    col_map = {
        "state": _find_col(cols, ["state"]),
        "commodity": _find_col(cols, ["commodity"]),
        "district": _find_col(cols, ["district"]),
        "market": _find_col(cols, ["market"]),
        "variety": _find_col(cols, ["variety"]),
        "grade": _find_col(cols, ["grade"]),
        "date": _find_col(cols, ["price date", "arrival_date", "arrival date", "date"]),
        "min": _find_col(cols, ["min_price", "min price", "min_x0020_price"]),
        "max": _find_col(cols, ["max_price", "max price", "max_x0020_price"]),
        "modal": _find_col(cols, ["modal_price", "modal price", "modal_x0020_price"]),
    }

    if col_map["date"]:
        df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
        df["Year"] = df[col_map["date"]].dt.year
        df["Month"] = df[col_map["date"]].dt.month
        df["Day"] = df[col_map["date"]].dt.day
        df["DayOfWeek"] = df[col_map["date"]].dt.dayofweek
        df["Quarter"] = df[col_map["date"]].dt.quarter
        df["WeekOfYear"] = pd.to_numeric(df[col_map["date"]].dt.isocalendar().week, errors="coerce")
        df["Season"] = df["Month"].fillna(1).astype(int).map(_season_from_month)
    else:
        for c in ["Year", "Month", "Day", "DayOfWeek", "Quarter", "WeekOfYear"]:
            df[c] = np.nan
        df["Season"] = "Rabi"

    if col_map["min"] and col_map["max"]:
        min_s = pd.to_numeric(df[col_map["min"]], errors="coerce")
        max_s = pd.to_numeric(df[col_map["max"]], errors="coerce")
        df["Min_Price"] = min_s
        df["Max_Price"] = max_s
        df["Price_Spread"] = max_s - min_s
        df["Price_Ratio"] = max_s / min_s.replace(0, np.nan)
    else:
        for c in ["Min_Price", "Max_Price", "Price_Spread", "Price_Ratio"]:
            df[c] = np.nan

    if col_map["modal"] and col_map["commodity"]:
        modal = pd.to_numeric(df[col_map["modal"]], errors="coerce")
        df["Modal_Price"] = modal
        if col_map["date"]:
            df = df.sort_values([col_map["commodity"], col_map["date"]])
        grp = df.groupby(col_map["commodity"], dropna=False)["Modal_Price"]
        df["Price_Lag1M"] = grp.shift(30)
        df["Price_RollMean7"] = grp.transform(lambda s: s.rolling(7, min_periods=1).mean())
    else:
        df["Modal_Price"] = np.nan
        df["Price_Lag1M"] = np.nan
        df["Price_RollMean7"] = np.nan

    return df, col_map


MODEL, SCALER, LABEL_ENCODERS, FEATURE_LIST = _load_artifacts()
DATA_DF, DATA_COLS = _load_data()

NUMERIC_DEFAULTS = {}
for col in FEATURE_LIST:
    if col in DATA_DF.columns:
        NUMERIC_DEFAULTS[col] = pd.to_numeric(DATA_DF[col], errors="coerce").median()
for col in FEATURE_LIST:
    if col not in NUMERIC_DEFAULTS or pd.isna(NUMERIC_DEFAULTS[col]):
        NUMERIC_DEFAULTS[col] = 0.0


def _encoder_key(token):
    token = token.lower()
    for key in LABEL_ENCODERS.keys():
        if token in str(key).lower():
            return key
    return None


ENCODER_KEYS = {
    "state": _encoder_key("state"),
    "district": _encoder_key("district"),
    "market": _encoder_key("market"),
    "commodity": _encoder_key("commodity"),
    "variety": _encoder_key("variety"),
    "grade": _encoder_key("grade"),
    "season": _encoder_key("season"),
}


def _encoder_lookup(key):
    if not key:
        return {}, []
    classes = [str(x) for x in LABEL_ENCODERS[key].classes_]
    mapping = {_canon(c).lower(): c for c in classes}
    clean = sorted({_canon(c) for c in classes if _canon(c) and _canon(c).lower() != "nan"}, key=str.lower)
    return mapping, clean


ENCODER_MAPS = {}
ENCODER_CLEAN = {}
for name, key in ENCODER_KEYS.items():
    mapping, clean = _encoder_lookup(key)
    ENCODER_MAPS[name] = mapping
    ENCODER_CLEAN[name] = clean


def _resolve_encoder_value(name, user_value):
    key = ENCODER_KEYS.get(name)
    if not key:
        return None
    classes = [str(x) for x in LABEL_ENCODERS[key].classes_]
    if not classes:
        return None
    wanted = _canon(user_value).lower()
    mapped = ENCODER_MAPS[name].get(wanted) if wanted else None
    if mapped is None:
        mapped = classes[0]
    return int(LABEL_ENCODERS[key].transform([mapped])[0]), mapped


def _safe_mean(series, fallback):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return float(fallback)


def _build_feature_row(payload):
    commodity = _canon(payload.get("commodity"))
    state = _canon(payload.get("state"))
    grade = _canon(payload.get("grade") or "FAQ")
    month = int(payload.get("month", datetime.now().month))
    year = int(payload.get("year", datetime.now().year))
    lag_price = float(payload.get("lag_price", 0) or 0)

    if lag_price <= 0:
        lag_price = _historical_mean(commodity, state, fallback=2500.0)

    row = dict(NUMERIC_DEFAULTS)

    for logical_name in ["state", "district", "market", "commodity", "variety", "grade"]:
        feat = ENCODER_KEYS.get(logical_name)
        if not feat:
            continue
        enc_col = f"{feat}_enc"
        if enc_col not in FEATURE_LIST:
            continue
        raw_val = payload.get(logical_name)
        if logical_name == "grade":
            raw_val = grade
        enc_val, _ = _resolve_encoder_value(logical_name, raw_val)
        row[enc_col] = enc_val

    if ENCODER_KEYS.get("season") and f"{ENCODER_KEYS['season']}_enc" in FEATURE_LIST:
        enc_val, _ = _resolve_encoder_value("season", _season_from_month(month))
        row[f"{ENCODER_KEYS['season']}_enc"] = enc_val

    time_values = {
        "Year": year,
        "Month": month,
        "Day": 15,
        "DayOfWeek": 2,
        "Quarter": ((month - 1) // 3) + 1,
        "WeekOfYear": max(1, min(53, int(round((month - 1) * 4.33 + 2)))),
    }
    for k, v in time_values.items():
        if k in FEATURE_LIST:
            row[k] = v

    if "Min_Price" in FEATURE_LIST:
        row["Min_Price"] = lag_price * 0.92
    if "Max_Price" in FEATURE_LIST:
        row["Max_Price"] = lag_price * 1.08
    if "Price_Spread" in FEATURE_LIST:
        row["Price_Spread"] = row.get("Max_Price", lag_price) - row.get("Min_Price", lag_price)
    if "Price_Ratio" in FEATURE_LIST:
        min_price = row.get("Min_Price", lag_price) or lag_price
        row["Price_Ratio"] = row.get("Max_Price", lag_price) / min_price if min_price else 1.0
    if "Price_Lag1M" in FEATURE_LIST:
        row["Price_Lag1M"] = lag_price
    if "Price_RollMean7" in FEATURE_LIST:
        row["Price_RollMean7"] = lag_price

    x = pd.DataFrame([row]).reindex(columns=FEATURE_LIST).fillna(0.0)
    return x, commodity, state, grade, month, year, lag_price


def _historical_filter(commodity, state):
    df = DATA_DF
    commodity_col = DATA_COLS.get("commodity")
    state_col = DATA_COLS.get("state")
    if commodity_col and commodity:
        df = df[df[commodity_col].astype(str).map(_canon).str.lower() == commodity.lower()]
    if state_col and state:
        state_df = df[df[state_col].astype(str).map(_canon).str.lower() == state.lower()]
        if not state_df.empty:
            df = state_df
    return df


def _historical_mean(commodity, state, fallback=2500.0):
    filtered = _historical_filter(commodity, state)
    if "Modal_Price" not in filtered.columns:
        return float(fallback)
    return _safe_mean(filtered["Modal_Price"], fallback)


def _year_count(df):
    if df.empty or "Year" not in df.columns:
        return 0
    years = pd.to_numeric(df["Year"], errors="coerce").dropna()
    if years.empty:
        return 0
    return int(years.nunique())


def _year_trend_frame(commodity, state, filtered_df):
    if _year_count(filtered_df) >= 2:
        return filtered_df

    commodity_df = _historical_filter(commodity, "")
    if _year_count(commodity_df) >= 2:
        return commodity_df

    state_df = _historical_filter("", state)
    if _year_count(state_df) >= 2:
        return state_df

    return DATA_DF


def _year_adjustment_factor(filtered_df, target_year):
    if filtered_df.empty or "Year" not in filtered_df.columns or "Modal_Price" not in filtered_df.columns:
        return 1.0

    years = pd.to_numeric(filtered_df["Year"], errors="coerce")
    prices = pd.to_numeric(filtered_df["Modal_Price"], errors="coerce")
    yearly = (
        pd.DataFrame({"Year": years, "Modal_Price": prices})
        .dropna(subset=["Year", "Modal_Price"])
        .groupby("Year", as_index=True)["Modal_Price"]
        .mean()
        .sort_index()
    )
    if yearly.empty:
        return 1.0

    ref_year = int(yearly.index.max())
    ref_price = float(yearly.loc[ref_year])
    if ref_price <= 0:
        return 1.0

    target_year = int(target_year)
    if target_year in yearly.index:
        raw_factor = float(yearly.loc[target_year]) / ref_price
    elif len(yearly) >= 2:
        recent = yearly.tail(3)
        x = recent.index.to_numpy(dtype=float)
        y = recent.to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        projected = float(slope * target_year + intercept)
        raw_factor = projected / ref_price
    else:
        raw_factor = 1.0

    return float(min(1.35, max(0.75, raw_factor)))


def _predict(payload):
    x, commodity, state, grade, month, year, lag_price = _build_feature_row(payload)
    filtered = _historical_filter(commodity, state)
    trend_df = _year_trend_frame(commodity, state, filtered)
    year_factor = _year_adjustment_factor(trend_df, year)
    model_name = type(MODEL).__name__.lower()
    if any(k in model_name for k in ["linear", "ridge"]):
        pred = float(MODEL.predict(SCALER.transform(x))[0])
    else:
        pred = float(MODEL.predict(x)[0])
    modal = max(0.0, round(pred * year_factor, 2))

    hist_mean = _historical_mean(commodity, state, fallback=modal)
    change_pct = 0.0 if hist_mean == 0 else ((modal - hist_mean) / hist_mean) * 100

    if "Modal_Price" in filtered.columns:
        std = float(pd.to_numeric(filtered["Modal_Price"], errors="coerce").std(skipna=True) or 0.0)
        sample_size = int(pd.to_numeric(filtered["Modal_Price"], errors="coerce").notna().sum())
    else:
        std = 0.0
        sample_size = 0
    band = max(60.0, modal * 0.06, std * 0.6)
    min_p = max(0.0, round(modal - band, 2))
    max_p = max(min_p, round(modal + band, 2))

    threshold = float(payload.get("threshold_pct", 5))
    if change_pct >= threshold:
        rec = "✅ SELL — Predicted price is above historical average."
        cls = "sell"
    elif change_pct <= -threshold:
        rec = "⏳ HOLD — Predicted price is below historical average."
        cls = "hold"
    else:
        rec = "⚖️ NEUTRAL — Price is near historical average."
        cls = "neutral"

    confidence = max(60.0, min(98.0, 70.0 + min(sample_size, 5000) / 200.0))
    payload_hash = hashlib.sha256(
        json.dumps(
            {
                "commodity": commodity,
                "state": state,
                "grade": grade,
                "month": month,
                "year": year,
                "lag_price": lag_price,
                "modal": modal,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "commodity": commodity,
        "state": state,
        "grade": grade,
        "month": month,
        "year": year,
        "lag_price": lag_price,
        "modal": modal,
        "min": min_p,
        "max": max_p,
        "historical_mean": round(hist_mean, 2),
        "change_pct": round(change_pct, 2),
        "recommendation": rec,
        "class": cls,
        "confidence": round(confidence, 1),
        "hash": payload_hash,
        "model_name": type(MODEL).__name__,
        "feature_count": len(FEATURE_LIST),
        "matched_rows": sample_size,
    }


def _forecast(payload):
    start_month = int(payload.get("start_month", datetime.now().month))
    start_year = int(payload.get("start_year", datetime.now().year))
    lag = float(payload.get("lag_price", 0) or 0)
    rows = []
    for i in range(12):
        month = ((start_month - 1 + i) % 12) + 1
        year = start_year + ((start_month - 1 + i) // 12)
        step_payload = dict(payload)
        step_payload["month"] = month
        step_payload["year"] = year
        if lag > 0:
            step_payload["lag_price"] = lag
        out = _predict(step_payload)
        lag = out["modal"]
        rows.append(
            {
                "month": month,
                "year": year,
                "label": datetime(year, month, 1).strftime("%b %y"),
                "modal": out["modal"],
                "min": out["min"],
                "max": out["max"],
                "change_pct": out["change_pct"],
                "class": out["class"],
            }
        )
    return {"rows": rows}


def _live_series(query):
    commodity = _canon(query.get("commodity", ["Tomato"])[0] if isinstance(query.get("commodity"), list) else query.get("commodity", "Tomato"))
    state = _canon(query.get("state", ["Karnataka"])[0] if isinstance(query.get("state"), list) else query.get("state", "Karnataka"))
    df = _historical_filter(commodity, state).copy()
    date_col = DATA_COLS.get("date")
    if date_col and "Modal_Price" in df.columns:
        df = df[df[date_col].notna() & pd.to_numeric(df["Modal_Price"], errors="coerce").notna()]
        df = df.sort_values(date_col)
        vals = pd.to_numeric(df["Modal_Price"], errors="coerce").dropna().tail(30).tolist()
    else:
        vals = []
    if not vals:
        base = _historical_mean(commodity, state, fallback=2500.0)
        vals = [round(base * (0.98 + (i / 400.0)), 2) for i in range(30)]
    return {"commodity": commodity, "state": state, "prices": vals}


def _ticker():
    if DATA_DF.empty or "Modal_Price" not in DATA_DF.columns:
        return {"items": []}
    c_col = DATA_COLS.get("commodity")
    if not c_col:
        return {"items": []}
    grp = DATA_DF.groupby(c_col, dropna=False)["Modal_Price"].mean().dropna().sort_values(ascending=False)
    items = []
    for idx, (name, val) in enumerate(grp.head(8).items()):
        change = 1.5 - (idx * 0.4)
        items.append(
            {
                "name": _canon(name),
                "price": round(float(val), 2),
                "change_pct": round(change, 2),
            }
        )
    return {"items": items}


def _metadata():
    return {
        "model_name": type(MODEL).__name__,
        "feature_count": len(FEATURE_LIST),
        "commodities": ENCODER_CLEAN.get("commodity", []),
        "states": ENCODER_CLEAN.get("state", []),
        "grades": ENCODER_CLEAN.get("grade", []),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, data, code=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/metadata":
            return self._json(_metadata())
        if parsed.path == "/api/ticker":
            return self._json(_ticker())
        if parsed.path == "/api/live-series":
            query = parse_qs(parsed.query)
            return self._json(_live_series(query))
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "Invalid JSON body"}, 400)

        try:
            if parsed.path == "/api/predict":
                return self._json(_predict(body))
            if parsed.path == "/api/forecast":
                return self._json(_forecast(body))
            return self._json({"error": "Unknown endpoint"}, 404)
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)


import os

def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving model-backed app at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
