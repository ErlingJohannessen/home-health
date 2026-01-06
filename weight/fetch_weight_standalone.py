# fetch_weight_standalone.py
# Standalone, non-interactive- HARD-CODED credentials (keep this file private!)
#  - OUTPUT_DIR controls where files are written
#  - All printouts captured in OUTPUT_DIR/weight_log.txt
#  - No GUI popups; figures are saved as JPG then program exits
# --------------------------------------------------------------------
import sys
import pathlib
import traceback
from datetime import date, timedelta, datetime

# ----------------- Output location (one constant) --------------------
# Change this to wherever yo/home/erling/garmin-output"
OUTPUT_DIR = pathlib.Path(r"./garmin_output").resolve()  # <-- edit if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / "weight_log.txt"

# ------------- Logging of all stdout/stderr to a file ---------------
# open in write mode, line-buffered; capture both stdout & stderr
sys.stdout = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
sys.stderr = sys.stdout
print(f"\n=== Run started {datetime.now().isoformat()} ===")
print(f"Output directory: {OUTPUT_DIR}")


# ----------------- Matplotlib non-interactive backend ----------------
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------- Garmin client & credentials -----------------------
from garminconnect import Garmin

# Replace with your actual Garmin credentials
EMAIL = "eaajohannessen@gmail.com"
PASSWORD = "Erlinga_22"

def iso(d): return d.isoformat()

def main():
    # ---------------------- Login (no prompts) -----------------------
    client = Garmin(EMAIL, PASSWORD)
    try:
        client.login()  # non-interactive; will fail if 2FA prompts are enforced
        print("Login: OK")
    except Exception as e:
        print(f"Login failed: {e}")
        traceback.print_exc()
        return 1

    # ---------------------- Date range -------------------------------
    to_date = date.today()
    from_date = to_date - timedelta(days=4 * 365)  # 4 years
    print(f"Date range: {from_date} .. {to_date}")

    # ---------------------- Retrieve weight rows ---------------------
    weight_rows = []
    try:
        rows = client.get_body_composition(iso(from_date), iso(to_date))
        weight_rows = rows.get("dateWeightList", [])
        print(f"get_body_composition rows: {len(weight_rows)}")
    except AttributeError:
        if hasattr(client, "get_weight"):
            weight_rows = client.get_weight(iso(from_date), iso(to_date)) or []
            print(f"get_weight rows: {len(weight_rows)}")
        elif hasattr(client, "get_daily_weight"):
            d = to_date
            print("Falling back to get_daily_weight per-day loop...")
            while d >= from_date:
                day_data = client.get_daily_weight(iso(d))
                if day_data:
                    if isinstance(day_data, list):
                        weight_rows.extend(day_data)
                    else:
                        weight_rows.append(day_data)
                d -= timedelta(days=1)
            print(f"get_daily_weight rows: {len(weight_rows)}")
        else:
            print("No weight API available in this garminconnect version.")
            weight_rows = []

    # ---------------------- Normalize & parse dates ------------------
    from datetime import datetime as dtclass

    def parse_any_date(s: str):
        if not s:
            return None
        try:
            return dtclass.fromisoformat(s)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return dtclass.strptime(s, fmt)
            except Exception:
                continue
        return None

    def extract_weight_kg(rec: dict):
        # Weight keys differ across forks
        w = rec.get("weight")
        if w is None:
            w = rec.get("weightInKilograms") or rec.get("weightInKg")
        if w is None:
            return None
        # Convert grams -> kg if values look like grams
        if isinstance(w, (int, float)) and w > 250:
            return float(w) / 1000.0
        return float(w)

    # Build dict by date to avoid duplicates (last measurement per day wins)
    by_date = {}
    for r in weight_rows:
        ts = r.get("calendarDate") or r.get("date") or r.get("measurementTime")
        dt = parse_any_date(ts)
        if not dt:
            continue
        day = dt.date()
        w_kg = extract_weight_kg(r)
        if w_kg is None:
            continue
        by_date[day] = w_kg

    # Actual measurement points (sorted)
    meas_dates = sorted(by_date.keys())
    meas_weights = [by_date[d] for d in meas_dates]

    if not meas_dates:
        print("No weight records found in the selected range.")
        client.session = None
        print("=== Run finished (no data) ===")
        return 0

    # ---------------------- Console summary --------------------------
    print("\nSamples (date → weight, days since previous):")
    prev_date = None
    for d in meas_dates:
        w = by_date[d]
        delta_days = 0 if prev_date is None else (d - prev_date).days
        print(f"{d}: {w:5.1f} kg (Δ {delta_days:3d} days)")
        prev_date = d
    days_since_last = (to_date - meas_dates[-1]).days
    print(f"\nDays since last sample: {days_since_last:3d} days\n")

    # ---------------------- Daily linear interpolation ----------------
    def build_daily_linear_interp(dates_list, weights_list):
        start = dates_list[0]
        end = dates_list[-1]
        total_days = (end - start).days
        daily_dates = [start + timedelta(days=i) for i in range(total_days + 1)]
        daily_weights = [None] * (total_days + 1)
        known_idx = {}
        for d, w in zip(dates_list, weights_list):
            idx = (d - start).days
            known_idx[idx] = w
            daily_weights[idx] = w
        idxs = sorted(known_idx.keys())
        for a, b in zip(idxs[:-1], idxs[1:]):
            wa, wb = daily_weights[a], daily_weights[b]
            span = b - a
            if span <= 0:
                continue
            slope = (wb - wa) / span
            for i in range(span + 1):  # inclusive ends
                daily_weights[a + i] = wa + slope * i
        return daily_dates, daily_weights

    daily_dates, daily_weights = build_daily_linear_interp(meas_dates, meas_weights)

    # ---------------------- Centered change per week ------------------
    def centered_change_per_week(weights, prev_len, next_len):
        change = [None] * len(weights)
        delta_days = (prev_len + next_len + 2) / 2.0  # distance between window centers in days
        scale = 7.0 / delta_days                      # kg per Δdays -> kg/week
        for i in range(len(weights)):
            if i >= prev_len and i + next_len < len(weights):
                prev = weights[i - prev_len:i]          # t-N_prev .. t-1
                foll = weights[i + 1:i + 1 + next_len]  # t+1 .. t+N_next
                if all(v is not None for v in prev + foll):
                    diff = (sum(foll) / next_len) - (sum(prev) / prev_len)
                    change[i] = diff * scale
        return change

    change_30d_per_week = centered_change_per_week(daily_weights, prev_len=15, next_len=15)
    change_week_per_week = centered_change_per_week(daily_weights, prev_len=4, next_len=3)

    # ---------------------- Subset: last 3 months ---------------------
    cutoff = to_date - timedelta(days=90)

    def first_ge(seq, target):
        for idx, d in enumerate(seq):
            if d >= target:
                return idx
        return len(seq)

    start_3m_idx = first_ge(daily_dates, cutoff)
    dates_3m = daily_dates[start_3m_idx:]
    weights_3m = daily_weights[start_3m_idx:]
    change_week_3m_per_week = change_week_per_week[start_3m_idx:]

    # Measurement samples within last 3 months (for solid-circle markers)
    meas_dates_3m = [d for d in meas_dates if d >= cutoff]
    meas_weights_3m = [by_date[d] for d in meas_dates_3m]

    # ---------------------- Plot 1: Full range ------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(daily_dates, daily_weights, linewidth=1.6, color="#1f77b4",
             label="Weight (kg) — interpolated daily")
    ax1.set_title("Weight over time (kg)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Weight (kg)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")

    ax1b = ax1.twinx()
    x_full = [d for d, c in zip(daily_dates, change_30d_per_week) if c is not None]
    y_full = [c for c in change_30d_per_week if c is not None]
    ax1b.plot(x_full, y_full, color="#ff7f0e", linewidth=2.0,
              label="Centered 30-day change (kg/week)")
    ax1b.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax1b.set_ylabel("Centered 30-day change (kg/week)")
    ax1b.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig1.tight_layout()

    # Save JPG to OUTPUT_DIR with good quality
    out1 = OUTPUT_DIR / "garmin_weight_full_centered_30day_change_per_week.jpg"
    fig1.savefig(out1, dpi=150, format="jpg",
                 pil_kwargs={"quality": 95, "optimize": True})
    plt.close(fig1)
    print(f"Saved: {out1}")

    # ---------------------- Plot 2: Last 3 months ---------------------
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    ax2.plot(dates_3m, weights_3m, linewidth=1.6, color="#1f77b4",
             label="Weight (kg) — interpolated daily")
    if meas_dates_3m:
        ax2.plot(
            meas_dates_3m, meas_weights_3m,
            linestyle="None", marker="o", markersize=6,
            markerfacecolor="#1f77b4", markeredgecolor="white", markeredgewidth=1.0,
            label="Samples", zorder=4
        )
    ax2.set_title("Weight (last 3 months)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Weight (kg)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")

    ax2b = ax2.twinx()
    x_3m = [d for d, c in zip(dates_3m, change_week_3m_per_week) if c is not None]
    y_3m = [c for c in change_week_3m_per_week if c is not None]
    ax2b.plot(x_3m, y_3m, color="#ff7f0e", linewidth=2.0,
              label="Centered weekly change (kg/week)")
    ax2b.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax2b.set_ylabel("Centered weekly change (kg/week)")
    ax2b.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    fig2.tight_layout()

    out2 = OUTPUT_DIR / "garmin_weight_3mo_centered_weekly_change_per_week_samples.jpg"
    fig2.savefig(out2, dpi=150, format="jpg",
                 pil_kwargs={"quality": 95, "optimize": True})
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ---------------------- Logout / cleanup -------------------------
    client.session = None  # clears the session object
    print("=== Run finished OK ===")
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
    # Explicit exit code for schedulers/cron
    sys.exit(exit_code)
