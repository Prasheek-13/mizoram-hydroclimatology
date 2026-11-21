import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
daily = pd.read_csv("mizoram_daily_rainfall_1901_2025.csv", parse_dates=["date"])
annual = pd.read_csv("mizoram_annual_rainfall_1901_2025.csv")

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month

# ---------------------------------------------------------
# 2. DATA QUALITY & ANOMALY CHECK
# ---------------------------------------------------------
print("=== 1. DATA QUALITY CHECK ===")
missing_days = daily["rainfall_mm"].isna().sum()
print(f"Total missing (NaN) days: {missing_days} out of {len(daily)}")

# Check the 5 driest years to see if dips are real or due to missing days
yearly_counts = daily.groupby("year")["rainfall_mm"].agg(
    valid_days="count", total_mm="sum", max_day_mm="max"
)
print("\n5 Driest Years (check valid_days for missing data):")
print(yearly_counts.nsmallest(5, "total_mm").to_string())

# ---------------------------------------------------------
# 3. SEASONAL BREAKDOWN (IMD Standard Seasons)
# ---------------------------------------------------------
def get_season(m):
    if m in [1, 2]:
        return "Winter (JF)"
    elif m in [3, 4, 5]:
        return "Pre-Monsoon (MAM)"
    elif m in [6, 7, 8, 9]:
        return "Monsoon (JJAS)"
    else:
        return "Post-Monsoon (OND)"

daily["season"] = daily["month"].apply(get_season)
seasonal = (
    daily.groupby(["year", "season"])["rainfall_mm"]
    .sum()
    .unstack(fill_value=0)
)
seasonal.to_csv("mizoram_seasonal_rainfall_1901_2025.csv")

# ---------------------------------------------------------
# 4. MANN-KENDALL TREND TEST & SEN'S SLOPE
# ---------------------------------------------------------
print("\n=== 2. MANN-KENDALL TREND TEST (1901–2025) ===")
for label, series in [
    ("Annual Total", annual["annual_total_mm"]),
    ("Monsoon (JJAS)", seasonal["Monsoon (JJAS)"]),
    ("Pre-Monsoon (MAM)", seasonal["Pre-Monsoon (MAM)"]),
    ("Max Daily Rainfall", annual["max_daily_mm"]),
]:
    res = mk.original_test(series.dropna())
    print(
        f"{label:20s} | Trend: {res.trend:12s} | "
        f"Sen's Slope: {res.slope:+.3f} mm/yr | p-value: {res.p:.4f}"
    )

# ---------------------------------------------------------
# 5. EXTREME VALUE ANALYSIS (GEV RETURN LEVELS)
# ---------------------------------------------------------
print("\n=== 3. EXTREME DAILY RAINFALL RETURN LEVELS (GEV) ===")
ams = annual["max_daily_mm"].dropna().values
shape_c, loc, scale = gev.fit(ams)

return_periods = [2, 5, 10, 25, 50, 100]
return_levels = {}
for T in return_periods:
    rl = gev.ppf(1 - 1 / T, shape_c, loc=loc, scale=scale)
    return_levels[T] = rl
    print(f"  {T:3d}-year return level: {rl:.1f} mm/day")

# ---------------------------------------------------------
# 6. SUMMARY PLOTS (SEASONAL + RETURN PERIOD CURVE)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot A: Monsoon vs Annual Rainfall
axes[0].plot(annual["year"], annual["annual_total_mm"], label="Annual", color="steelblue", alpha=0.8)
axes[0].plot(seasonal.index, seasonal["Monsoon (JJAS)"], label="Monsoon (JJAS)", color="darkorange", alpha=0.9)
axes[0].set_title("Annual vs. Monsoon Rainfall (1901–2025)")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Rainfall (mm)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot B: GEV Return Level Curve
T_grid = np.logspace(0.05, 2.1, 200)
rl_grid = gev.ppf(1 - 1 / T_grid, shape_c, loc=loc, scale=scale)
axes[1].plot(T_grid, rl_grid, color="crimson", lw=2, label="GEV Fit")
axes[1].scatter(return_periods, [return_levels[T] for T in return_periods], color="black", zorder=5)
axes[1].set_xscale("log")
axes[1].set_title("Extreme Daily Rainfall Return Levels (GEV)")
axes[1].set_xlabel("Return Period (Years, log scale)")
axes[1].set_ylabel("1-Day Max Rainfall (mm)")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("mizoram_hydroclimatology_summary.png", dpi=300)
plt.show()