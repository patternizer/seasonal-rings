#!/usr/bin/env python3
"""
Radial climatological rings chart for monthly data.

Features
- Concentric rings: inner = latest year (e.g., 2025), then 2024, ... to earliest (or user-limited) year
- 12 segments per ring aligned like a clock with *December at 12 o'clock*, January at 1 o'clock, etc.
- Central hole with a large bold label (default: "SST")
- Generic CLI: pass CSV path, data (value) column, year column, and month column
- Colorbar with a sensible number of divisions (default: 10)
- Handles missing months (rendered with a neutral color)

Example
-------
python radial_climo_rings.py \
  --data /path/to/HadSST.4.2.0.0_monthly_GLOBE.csv \
  --value-col anomaly --year-col year --month-col month \
  --center-label SST --title "Global SST anomaly (HadSST4)"

Notes
-----
- Month column can be integer [1..12] or strings ("Jan", "January"). Use --month-format if parsing fails.
- By default, color scaling is symmetric around 0 (good for anomalies). Use --no-symmetric or specify --vmin/--vmax to override.
- You can limit the plotted year range with --year-min and/or --year-max.
"""
import argparse
import math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import MaxNLocator


# ----------------------
# Parsing helpers
# ----------------------

MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
MONTH_FULL = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}


def parse_month(x, month_format: str = "auto") -> int:
    """Parse a month value into 1..12.

    month_format: 'auto' | 'abbr' | 'full' | 'number'
    """
    if pd.isna(x):
        return np.nan
    if month_format == 'number':
        try:
            m = int(x)
            return m if 1 <= m <= 12 else np.nan
        except Exception:
            return np.nan
    if month_format in ('abbr', 'full'):
        s = str(x).strip().lower()
        table = MONTH_ABBR if month_format == 'abbr' else MONTH_FULL
        return table.get(s, np.nan)

    # auto
    try:
        m = int(x)
        if 1 <= m <= 12:
            return m
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in MONTH_ABBR:
        return MONTH_ABBR[s]
    if s in MONTH_FULL:
        return MONTH_FULL[s]
    # last resort: try pandas datetime
    try:
        dt = pd.to_datetime(s, format='%b', errors='coerce')
        if pd.notna(dt):
            return int(dt.month)
        dt = pd.to_datetime(s, format='%B', errors='coerce')
        if pd.notna(dt):
            return int(dt.month)
    except Exception:
        pass
    return np.nan


# ----------------------
# Plot logic
# ----------------------

# Position mapping so that December is at 12 o'clock (index 0),
# January at 1 o'clock (index 1), ..., November at 11 o'clock (index 11)
MONTH_TO_POS = {12: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}


def build_value_map(df: pd.DataFrame, year_col: str, month_col: str, value_col: str) -> Dict[Tuple[int, int], float]:
    vals = {}
    for _, row in df.iterrows():
        y = int(row[year_col])
        m = int(row[month_col])
        vals[(y, m)] = row[value_col]
    return vals


def compute_norm(values: np.ndarray,
                 symmetric: bool,
                 robust: bool,
                 vmin: Optional[float],
                 vmax: Optional[float],
                 quantiles: Tuple[float, float] = (2, 98)):
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        # fallback
        return Normalize(vmin=-1, vmax=1)

    if vmin is not None and vmax is not None:
        if symmetric:
            center = 0.0
            return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            return Normalize(vmin=vmin, vmax=vmax)

    if robust:
        low, high = np.nanpercentile(vals, quantiles)
    else:
        low, high = float(np.nanmin(vals)), float(np.nanmax(vals))

    if symmetric:
        bound = max(abs(low), abs(high))
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    else:
        return Normalize(vmin=low, vmax=high)


def draw_radial_rings(
    df: pd.DataFrame,
    value_col: str,
    year_col: str,
    month_col: str,
    center_label: str = "SST",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    figsize: float = 8.0,
    inner_radius: float = 0.55,
    ring_width: float = 0.28,
    ring_gap: float = 0.02,
    cmap: str = 'coolwarm',
    symmetric: bool = True,
    robust: bool = True,
    quantiles: Tuple[float, float] = (2, 98),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    show_year_labels: bool = True,
    cbar_bins: int = 10,
    missing_color: str = '#d9d9d9',
    dpi: int = 300,
    save: Optional[str] = None,
):
    # Filter year range
    years_all = sorted(df[year_col].dropna().astype(int).unique())
    if year_min is None:
        year_min = min(years_all)
    if year_max is None:
        year_max = max(years_all)

    years = [y for y in years_all if year_min <= y <= year_max]
    if not years:
        raise ValueError("No data within the specified year range.")

    # inner ring should be latest year
    years_sorted = sorted(years, reverse=True)
    n_rings = len(years_sorted)

    # Value lookup map
    val_map = build_value_map(df, year_col, month_col, value_col)

    # Gather values for norm
    all_vals = []
    for key, v in val_map.items():
        if np.isfinite(v):
            all_vals.append(float(v))
    all_vals = np.array(all_vals, dtype=float) if all_vals else np.array([0.0])

    norm = compute_norm(all_vals, symmetric=symmetric, robust=robust, vmin=vmin, vmax=vmax, quantiles=quantiles)
    cmap_obj = plt.get_cmap(cmap).copy()
    try:
        cmap_obj.set_bad(missing_color)
    except Exception:
        pass

    # Figure & axis
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')

    # Compute maximum radius to set limits
    R_max = inner_radius + n_rings * ring_width + (n_rings - 1) * ring_gap
    ax.set_xlim(-R_max - 0.2, R_max + 0.2)
    ax.set_ylim(-R_max - 0.2, R_max + 0.2)

    # Draw rings
    for i, year in enumerate(years_sorted):
        r_in = inner_radius + i * (ring_width + ring_gap)
        r_out = r_in + ring_width
        # 12 months
        for month in range(1, 13):
            pos = MONTH_TO_POS[month]
            theta_center = 90 - 30 * pos  # degrees; 90° is 12 o'clock; clockwise is negative direction
            theta1 = theta_center - 15
            theta2 = theta_center + 15

            val = val_map.get((year, month), np.nan)
            color = cmap_obj(norm(val)) if np.isfinite(val) else missing_color

            wedge = Wedge(center=(0, 0), r=r_out, theta1=theta1, theta2=theta2, width=ring_width, facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(wedge)

        if show_year_labels:
            # Place year label at 9 o'clock, centered radially in the ring
            r_mid = 0.5 * (r_in + r_out)
            ax.text(-r_mid, 0, str(year), va='center', ha='right', fontsize=8)

    # Central hole + label
    hole = Circle((0, 0), radius=inner_radius - 0.01, facecolor='white', edgecolor='none', zorder=10)
    ax.add_patch(hole)
    ax.text(0, 0, center_label, ha='center', va='center', fontsize=24, fontweight='bold')

    # Colorbar with sensible divisions
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.locator = MaxNLocator(nbins=cbar_bins)
    cbar.update_ticks()
    cbar.ax.set_ylabel(value_col, rotation=270, va='bottom')

    if title:
        ax.set_title(title, pad=14)

    if save:
        fig.savefig(save, bbox_inches='tight', dpi=dpi)
    else:
        plt.show()


# ----------------------
# CLI
# ----------------------

def main():
    p = argparse.ArgumentParser(description="Concentric radial rings chart for monthly data (December at 12 o'clock).")
    p.add_argument('--data', required=True, help='Path to CSV file with monthly data')
    p.add_argument('--value-col', required=True, help='Column containing the numeric monthly values (e.g., anomaly)')
    p.add_argument('--year-col', default='year', help='Column with year (integer)')
    p.add_argument('--month-col', default='month', help='Column with month (1..12 or month names)')
    p.add_argument('--month-format', default='auto', choices=['auto', 'abbr', 'full', 'number'], help='Hint for parsing month values if not numeric')

    p.add_argument('--center-label', default='SST', help='Text label inside the central hole')
    p.add_argument('--title', default=None, help='Optional chart title')
    p.add_argument('--year-min', type=int, default=None, help='Earliest year to include')
    p.add_argument('--year-max', type=int, default=None, help='Latest year to include')

    p.add_argument('--figsize', type=float, default=8.0, help='Figure size (inches, square)')
    p.add_argument('--inner-radius', type=float, default=0.55, help='Radius of the central hole')
    p.add_argument('--ring-width', type=float, default=0.28, help='Radial width of each year ring')
    p.add_argument('--ring-gap', type=float, default=0.02, help='Gap between rings')

    p.add_argument('--cmap', default='coolwarm', help='Matplotlib colormap name (diverging recommended for anomalies)')
    p.add_argument('--symmetric', dest='symmetric', action='store_true', help='Color scale symmetric around 0 (default)')
    p.add_argument('--no-symmetric', dest='symmetric', action='store_false', help='Disable symmetric color scaling around 0')
    p.set_defaults(symmetric=True)

    p.add_argument('--robust', dest='robust', action='store_true', help='Use robust quantiles for vmin/vmax (default)')
    p.add_argument('--no-robust', dest='robust', action='store_false', help='Use min/max for vmin/vmax')
    p.set_defaults(robust=True)

    p.add_argument('--quantiles', default='2,98', help='Quantiles for robust scaling as "low,high" (e.g., 2,98)')
    p.add_argument('--vmin', type=float, default=None, help='Force color scale minimum')
    p.add_argument('--vmax', type=float, default=None, help='Force color scale maximum')
    p.add_argument('--cbar-bins', type=int, default=10, help='Approximate number of tick divisions on the colorbar')

    p.add_argument('--show-year-labels', dest='show_year_labels', action='store_true', help='Show year labels at 9 o\'clock (default)')
    p.add_argument('--no-year-labels', dest='show_year_labels', action='store_false', help='Hide year labels')
    p.set_defaults(show_year_labels=True)

    p.add_argument('--save', default=None, help='Path to save the figure instead of displaying (e.g., out.png)')
    p.add_argument('--dpi', type=int, default=300, help='Figure DPI')

    args = p.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.year_col not in df.columns:
        raise ValueError(f"Year column '{args.year_col}' not found in CSV.")
    if args.month_col not in df.columns:
        raise ValueError(f"Month column '{args.month_col}' not found in CSV.")
    if args.value_col not in df.columns:
        raise ValueError(f"Value column '{args.value_col}' not found in CSV.")

    # Coerce year
    df[args.year_col] = pd.to_numeric(df[args.year_col], errors='coerce').astype('Int64')

    # Coerce month
    if pd.api.types.is_numeric_dtype(df[args.month_col]):
        df[args.month_col] = pd.to_numeric(df[args.month_col], errors='coerce').astype('Int64')
    else:
        df[args.month_col] = df[args.month_col].apply(lambda x: parse_month(x, args.month_format)).astype('Int64')

    # Coerce value
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors='coerce')

    # Drop rows with missing year/month
    df = df.dropna(subset=[args.year_col, args.month_col])
    df = df[(df[args.month_col] >= 1) & (df[args.month_col] <= 12)]

    # Parse quantiles
    try:
        q_low, q_high = [float(q.strip()) for q in args.quantiles.split(',')]
    except Exception:
        q_low, q_high = 2.0, 98.0

    draw_radial_rings(
        df=df,
        value_col=args.value_col,
        year_col=args.year_col,
        month_col=args.month_col,
        center_label=args.center_label,
        year_min=args.year_min,
        year_max=args.year_max,
        figsize=args.figsize,
        inner_radius=args.inner_radius,
        ring_width=args.ring_width,
        ring_gap=args.ring_gap,
        cmap=args.cmap,
        symmetric=args.symmetric,
        robust=args.robust,
        quantiles=(q_low, q_high),
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        show_year_labels=args.show_year_labels,
        cbar_bins=args.cbar_bins,
        save=args.save,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
