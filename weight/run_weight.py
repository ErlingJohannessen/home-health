#!/usr/bin/env python3
"""
run_weight.py
Non-interactive; robust token handling for garminconnect + garth (via garminconnect).

Key behavior:
  - Use tokenstore for login so tokens are re-used and refreshed automatically.
  - If token refresh fails with 401 Unauthorized, clear tokenstore and retry once.
  - If rate-limited (429 Too Many Requests), fail fast to avoid hammering Garmin.
  - Do NOT hardcode credentials; use env vars:
        GARMIN_EMAIL, GARMIN_PASSWORD
  - Logs to OUTPUT_DIR/weight_log.txt and saves a weight plot as JPG.

Working directory:
  - Script expects to run from the directory that contains ./garmin_output (your run_weight.sh cd's to it). [1](https://masterx-my.sharepoint.com/personal/erling_johannessen_km_kongsberg_com/Documents/Microsoft%20Copilot%20Chat%20Files/run_weight.py)
"""

import os
import sys
import json
import shutil
import pathlib
import traceback
from datetime import date, timedelta, datetime, timezone

# ----------------- Output location --------------------
OUTPUT_DIR = pathlib.Path("./garmin_output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / "weight_log.txt"

# Token store directory (used by garminconnect/garth)
TOKEN_DIR = OUTPUT_DIR / "garmin_token"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

# Optional: files that may exist in token dir (we don't strictly depend on them)
OAUTH1 = TOKEN_DIR / "oauth1_token.json"
OAUTH2 = TOKEN_DIR / "oauth2_token.json"

COOLDOWN_FILE = OUTPUT_DIR / "last_429.txt"
COOLDOWN_HOURS = 6

# ----------------- Logging ----------------------------
sys.stdout = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
sys.stderr = sys.stdout

print(f"\n=== Run started {datetime.now().isoformat()} ===")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Token store dir: {TOKEN_DIR}")

# ----------------- Matplotlib (headless) --------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------- Garmin client ----------------------
from garminconnect import Garmin


def iso(d: date) -> str:
    return d.isoformat()


def _clear_token_dir() -> None:
    """Remove tokenstore directory contents to force a clean login next time."""
    try:
        if TOKEN_DIR.exists():
            shutil.rmtree(TOKEN_DIR)
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        print("Cleared token directory.")
    except Exception as e:
        print(f"Could not clear token directory: {e}")

def in_cooldown():
    if not COOLDOWN_FILE.exists():
        return False
    try:
        ts = datetime.fromisoformat(COOLDOWN_FILE.read_text().strip())
        return datetime.utcnow() < ts + timedelta(hours=COOLDOWN_HOURS)
    except Exception:
        return False

def mark_429():
    COOLDOWN_FILE.write_text(datetime.utcnow().isoformat())

def login_tokenstore_or_password_seed(email, password):
    """
    Works with garminconnect 0.2.38 behavior:
      - If oauth1_token.json exists: load/refresh via tokenstore
      - If missing: do password login once, then save tokens to tokenstore
      - If 401: clear tokenstore and retry once
      - If 429: fail fast (avoid hammering Garmin)
    """

    
    if in_cooldown():
        print("In Garmin cooldown window — skipping login attempt.")
        return None

    client = Garmin(email, password)

    oauth1_path = TOKEN_DIR / "oauth1_token.json"

    # 1) If token file exists, try tokenstore login
    if oauth1_path.exists():
        try:
            client.login(tokenstore=str(TOKEN_DIR))
            print("Login OK (tokenstore).")
            return client
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                print(f"Rate-limited by Garmin (429). Will try next scheduled run. Details: {e}")
                
                mark_429()
                print("Rate limited (429). Entering cooldown.")
                return None

            if "401" in msg or "Unauthorized" in msg:
                print(f"Token login failed with 401. Clearing tokenstore and retrying once. Details: {e}")
                _clear_token_dir()
                try:
                    client.login()  # password login
                    print("Password login OK after token clear.")
                    # Save tokens for next run
                    client.garth.save(str(TOKEN_DIR))
                    print("Saved tokens to tokenstore.")
                    return client
                except Exception as e2:
                    print(f"Login still failed after token clear: {e2}")
                    traceback.print_exc()
                    return None

            print(f"Tokenstore login failed: {e}")
            traceback.print_exc()
            return None

    # 2) Token file missing -> do password login and then save tokens
    try:
        print("Token file missing; performing password login to seed tokenstore...")
        client.login()
        print("Password login OK.")
        client.garth.save(str(TOKEN_DIR))  # IMPORTANT: create oauth1_token.json
        print("Saved tokens to tokenstore.")
        return client
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            print(f"Rate-limited by Garmin (429). Will try next scheduled run. Details: {e}")
            return None
        print(f"Password login failed: {e}")
        traceback.print_exc()
        return None





def _extract_weight_rows(client, from_date: date, to_date: date):
    """
    Try to fetch weight/body composition history across different garminconnect methods.
    Returns a list of rows with at least: timestamp/date + weight (kg).
    """
    # Preferred (commonly available):
    # get_body_composition(from, to) returns dict with "dateWeightList"
    try:
        data = client.get_body_composition(iso(from_date), iso(to_date))
        rows = data.get("dateWeightList", []) if isinstance(data, dict) else []
        print(f"get_body_composition rows: {len(rows)}")
        if rows:
            return rows
    except Exception as e:
        print(f"get_body_composition failed: {e}")

    # Fallbacks vary by version:
    if hasattr(client, "get_weight"):
        try:
            rows = client.get_weight(iso(from_date), iso(to_date)) or []
            print(f"get_weight rows: {len(rows)}")
            if rows:
                return rows
        except Exception as e:
            print(f"get_weight failed: {e}")

    # Very defensive: daily endpoint fallback (slow)
    if hasattr(client, "get_daily_weight"):
        try:
            rows = []
            d = from_date
            while d <= to_date:
                day = client.get_daily_weight(iso(d))
                # Normalize into a row-like dict if possible
                if day:
                    rows.append(day)
                d += timedelta(days=1)
            print(f"get_daily_weight rows: {len(rows)}")
            if rows:
                return rows
        except Exception as e:
            print(f"get_daily_weight failed: {e}")

    return []


def _normalize_rows_to_series(rows):
    """
    Convert Garmin rows into two parallel arrays:
      - dates (datetime.date)
      - weights_kg (float)
    Handles common shapes seen in dateWeightList.
    """
    dates = []
    weights = []

    for r in rows:
        if not isinstance(r, dict):
            continue

        # Common keys in "dateWeightList":
        # - "date" can be millis, iso string, or yyyymmdd-ish
        # - "weight" or "weightValue" may appear, sometimes grams
        dt = None

        # 1) Try epoch millis
        if "date" in r and isinstance(r["date"], (int, float)):
            try:
                dt = datetime.fromtimestamp(r["date"] / 1000, tz=timezone.utc).date()
            except Exception:
                dt = None

        # 2) Try ISO-like string
        if dt is None and "date" in r and isinstance(r["date"], str):
            s = r["date"].strip()
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    dt = datetime.strptime(s[:10], fmt).date()
                    break
                except Exception:
                    pass

        # 3) Sometimes "calendarDate" or similar
        if dt is None:
            for k in ("calendarDate", "dateTimeLocal", "dateTimeUtc"):
                v = r.get(k)
                if isinstance(v, str) and len(v) >= 10:
                    try:
                        dt = datetime.strptime(v[:10], "%Y-%m-%d").date()
                        break
                    except Exception:
                        continue

        # Weight extraction
        w = None
        for k in ("weight", "weightValue", "weightInGrams", "weightInKg"):
            if k in r and isinstance(r[k], (int, float)):
                w = float(r[k])
                # Heuristics: if grams, convert to kg
                if k == "weightInGrams" or w > 250:  # 80kg would be 80000 grams; 80000 > 250
                    # If values look like grams (e.g., 80000), convert
                    if w > 1000:
                        w = w / 1000.0
                break

        # Sometimes stored in nested structures
        if w is None:
            v = r.get("weight")
            if isinstance(v, dict) and "value" in v:
                try:
                    w = float(v["value"])
                except Exception:
                    w = None

        if dt is not None and w is not None:
            dates.append(dt)
            weights.append(w)

    # Sort by date
    if dates and weights:
        pairs = sorted(zip(dates, weights), key=lambda x: x[0])
        dates, weights = zip(*pairs)
        return list(dates), list(weights)

    return [], []


def _save_plot(dates, weights):
    if not dates or not weights:
        print("No data to plot.")
        return

    # Convert date objects to datetimes for matplotlib
    x = [datetime(d.year, d.month, d.day) for d in dates]
    y = weights

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_title("Weight (kg)")
    ax.set_ylabel("kg")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    out = OUTPUT_DIR / "weight_timeseries.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def _save_csv(dates, weights):
    if not dates or not weights:
        return

    out = OUTPUT_DIR / "weight_timeseries.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("date,weight_kg\n")
        for d, w in zip(dates, weights):
            f.write(f"{d.isoformat()},{w:.2f}\n")
    print(f"Saved CSV: {out}")


def main():
    # Require env vars (safer than defaults)
    email = "eaajohannessen@gmail.com"  #os.environ.get("GARMIN_EMAIL")
    password = "Erlinga_22"             #os.environ.get("GARMIN_PASSWORD")
    if not email or not password:
        print("ERROR: GARMIN_EMAIL and GARMIN_PASSWORD must be set in the environment.")
        return 2

    client = login_tokenstore_or_password_seed(email, password)
    if client is None:
        return 1

    # Date range (4 years like your original intent) [1](https://masterx-my.sharepoint.com/personal/erling_johannessen_km_kongsberg_com/Documents/Microsoft%20Copilot%20Chat%20Files/run_weight.py)
    to_date = date.today()
    from_date = to_date - timedelta(days=4 * 365)
    print(f"Date range: {from_date} .. {to_date}")

    # Retrieve rows
    rows = _extract_weight_rows(client, from_date, to_date)
    print(f"Total raw rows: {len(rows)}")

    # Normalize to series
    dates, weights = _normalize_rows_to_series(rows)
    print(f"Normalized points: {len(dates)}")

    # Save outputs
    _save_csv(dates, weights)
    _save_plot(dates, weights)

    print("Done.")
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stdout.close()
        except Exception:
            pass
    sys.exit(exit_code)
