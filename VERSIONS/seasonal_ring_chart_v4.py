#!/usr/bin/env python3
"""
Radial climatological rings chart for monthly data (improved).

Adds:
- --cbar-shrink to shorten colorbar length.
- --year-axis-angle-deg to position the year axis at any angle (deg).
- --cbar-step for exact colorbar step spacing (e.g., 0.1).
- Year labels centered on tick marks; labels offset along the axis-normal.
- Headless rendering (Agg), save-only.

Angle convention: 12 o'clock = 90°, 3 o'clock = 0°, angles decrease clockwise.
"""
import argparse
import math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter

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
MONTH_ABBR_CAPS = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",
                   8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}

def parse_month(x, month_format: str = "auto") -> int:
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

# December at 12 o'clock (index 0), January at 1 o'clock (index 1), ...
MONTH_TO_POS = {12: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}

def build_value_map(df: pd.DataFrame, year_col: str, month_col: str, value_col: str) -> Dict[Tuple[int, int], float]:
    vals = {}
    for _, row in df.iterrows():
        y = int(row[year_col])
        m = int(row[month_col])
        vals[(y, m)] = row[value_col]
    return vals

def draw_radial_rings(
    df: pd.DataFrame,
    value_col: str,
    year_col: str,
    month_col: str,
    center_label: Optional[str] = None,
    hide_center_label: bool = False,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    figsize: float = 7.5,
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
    cbar_segments: int = 10,
    cbar_extend: str = 'both',
    year_label_fontsize: int = 5,
    year_label_step: Optional[int] = None,
    max_year_labels: int = 12,
    year_tick_length: float = 0.04,
    cbar_fraction: float = 0.045,   # thickness
    cbar_shrink: float = 1.0,       # length
    cbar_pad: float = 0.03,
    center_label_fontsize: int = 18,
    title_fontsize: int = 12,
    cbar_tick_fontsize: int = 9,
    cbar_label_fontsize: int = 10,
    year_label_offset: float = 0.0,  # along axis-normal
    add_month_labels: bool = True,
    month_label_fontsize: int = 10,
    month_label_offset: float = 0.08,
    missing_color: str = '#d9d9d9',
    year_axis_angle_deg: float = 180.0,  # default along negative x-axis
    year_axis_linewidth: float = 0.9,
    year_tick_width: float = 0.9,
    cbar_step: Optional[float] = None,   # e.g., 0.1 for exact steps & ticks
    dpi: int = 600,
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
    years_sorted = sorted(years, reverse=True)
    n_rings = len(years_sorted)

    # Determine year label step
    if show_year_labels:
        if year_label_step is None:
            candidates = [1, 2, 5, 10, 20, 25, 50, 100]
            max_labels = max(1, max_year_labels)
            step = candidates[-1]
            for s in candidates:
                if math.ceil(n_rings / s) <= max_labels:
                    step = s
                    break
        else:
            step = max(1, int(year_label_step))
    else:
        step = None

    # Value lookup map and range
    val_map = build_value_map(df, year_col, month_col, value_col)
    vals = np.array([float(v) for v in val_map.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        vals = np.array([0.0])

    if (vmin is None) or (vmax is None):
        low, high = (np.nanpercentile(vals, quantiles) if robust
                     else (float(np.nanmin(vals)), float(np.nanmax(vals))))
        if symmetric:
            bound = max(abs(low), abs(high))
            vmin_use, vmax_use = -bound, bound
        else:
            vmin_use, vmax_use = low, high
    else:
        vmin_use, vmax_use = vmin, vmax

    # Discrete levels (optionally exact step)
    if cbar_step is not None and cbar_step > 0:
        if symmetric:
            bound = max(abs(vmin_use), abs(vmax_use))
            bound = np.ceil(bound / cbar_step) * cbar_step
            vmin_use, vmax_use = -bound, bound
        else:
            vmin_use = np.floor(vmin_use / cbar_step) * cbar_step
            vmax_use = np.ceil(vmax_use / cbar_step) * cbar_step
        levels = np.arange(vmin_use, vmax_use + 0.5*cbar_step, cbar_step)
    else:
        levels = np.linspace(vmin_use, vmax_use, cbar_segments + 1)

    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, len(levels)-1))
    cmap_obj = ListedColormap(colors)
    try:
        cmap_obj.set_bad(missing_color)
    except Exception:
        pass
    norm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)

    # Figure & axis
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')

    # Geometry
    R_max = inner_radius + n_rings * ring_width + (n_rings - 1) * ring_gap
    ax.set_xlim(-R_max - 0.35, R_max + 0.35)
    ax.set_ylim(-R_max - 0.35, R_max + 0.35)

    # Rings
    for i, year in enumerate(years_sorted):
        r_in = inner_radius + i * (ring_width + ring_gap)
        r_out = r_in + ring_width
        for month in range(1, 13):
            pos = MONTH_TO_POS[month]
            theta_center = 90 - 30 * pos
            theta1 = theta_center - 15
            theta2 = theta_center + 15
            val = val_map.get((year, month), np.nan)
            color = sm.to_rgba(val) if np.isfinite(val) else missing_color
            wedge = Wedge(center=(0, 0), r=r_out, theta1=theta1, theta2=theta2,
                          width=ring_width, facecolor=color, edgecolor='white', linewidth=0.5, zorder=1)
            ax.add_patch(wedge)

        if show_year_labels and (step is not None) and (i % step == 0):
            r_mid = 0.5 * (r_in + r_out)
            theta = np.deg2rad(year_axis_angle_deg)
            # normal to the axis (for offset and tick orientation)
            nx, ny = -np.sin(theta), np.cos(theta)
            x = r_mid * np.cos(theta) + year_label_offset * nx
            y = r_mid * np.sin(theta) + year_label_offset * ny            
            ax.text(x, y, str(year), va='center', ha='center', fontsize=year_label_fontsize, fontweight='bold', zorder=200, clip_on=False)
        
    # Central hole + (optional) label
    hole = Circle((0, 0), radius=inner_radius - 0.01, facecolor='white', edgecolor='none', zorder=2)
    ax.add_patch(hole)
    if (not hide_center_label) and center_label:
        ax.text(0, 0, center_label, ha='center', va='center',
                fontsize=center_label_fontsize, fontweight='bold')

    # Year axis line at requested angle
    theta = np.deg2rad(year_axis_angle_deg)
    x1, y1 = (inner_radius + ring_width) * np.cos(theta), (inner_radius + ring_width) * np.sin(theta)
    x2, y2 = R_max * np.cos(theta), R_max * np.sin(theta)
    ax.plot([x1, x2], [y1, y2], color='black', linewidth=year_axis_linewidth, zorder=150)
    
    # Tick marks along axis for labeled rings
    if show_year_labels:
        for i, year in enumerate(years_sorted):
            label_this = (i % (year_label_step if year_label_step else step) == 0)
            if label_this:
                r_in_i = inner_radius + i * (ring_width + ring_gap)
                r_out_i = r_in_i + ring_width
                r_mid_i = 0.5 * (r_in_i + r_out_i)
                # axis point
                px = r_mid_i * np.cos(theta)
                py = r_mid_i * np.sin(theta)
                # normal for tick direction
                nx, ny = -np.sin(theta), np.cos(theta)
                dx, dy = (year_tick_length/2) * nx, (year_tick_length/2) * ny
                ax.plot([px - dx, px + dx], [py - dy, py + dy], 
                        color='black', linewidth=year_tick_width, zorder=150)
                        
    # Month labels around outside
    if add_month_labels:
        r_lbl = R_max + month_label_offset
        for month in range(1, 13):
            pos = MONTH_TO_POS[month]
            theta_deg = 90 - 30 * pos
            t = np.deg2rad(theta_deg)
            x = r_lbl * np.cos(t)
            y = r_lbl * np.sin(t)
            ax.text(x, y, MONTH_ABBR_CAPS[month], ha='center', va='center',
                    fontsize=month_label_fontsize, fontweight='bold',
                    color='black', clip_on=False, zorder=200)

    # Colorbar
    sm.set_array([])
    if cbar_step is not None and cbar_step > 0:
        ticks = np.arange(levels[0], levels[-1] + 0.5*cbar_step, cbar_step)
    else:
        # thin boundary ticks to about cbar_bins
        n_edges = len(levels)
        tick_step = max(1, int(np.ceil((n_edges - 1) / max(1, cbar_bins))))
        ticks = levels[::tick_step]
        if ticks[-1] != levels[-1]:
            ticks = np.append(ticks, levels[-1])

    cbar = fig.colorbar(sm, ax=ax,
                        fraction=cbar_fraction, pad=cbar_pad,
                        boundaries=levels, spacing='proportional',
                        extend=cbar_extend, ticks=ticks,
                        drawedges=True, shrink=cbar_shrink)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar.ax.set_ylabel(value_col, rotation=270, va='bottom', fontsize=cbar_label_fontsize)
    # Format to 0.1 precision if step provided
    if cbar_step is not None and cbar_step > 0:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if title:
        ax.set_title(title, pad=14, fontsize=title_fontsize)

    out_path = save if save else f"seasonal_ring_chart_{value_col}_{years_sorted[-1]}-{years_sorted[0]}.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return out_path

# ----------------------
# CLI
# ----------------------

def main():
    from matplotlib.colors import TwoSlopeNorm, Normalize  # kept for compatibility (not directly used here)

    p = argparse.ArgumentParser(description="Concentric radial rings chart for monthly data (December at 12 o'clock).")
    p.add_argument('--data', required=True, help='Path to CSV file with monthly data')
    p.add_argument('--value-col', required=True, help='Column containing the numeric monthly values (e.g., anomaly)')
    p.add_argument('--year-col', default='year', help='Column with year (integer)')
    p.add_argument('--month-col', default='month', help='Column with month (1..12 or month names)')
    p.add_argument('--month-format', default='auto', choices=['auto', 'abbr', 'full', 'number'])

    p.add_argument('--center-label', default=None, help='Label inside the central hole (omit or empty to hide)')
    p.add_argument('--hide-center-label', action='store_true', help='Do not draw label in the center')
    p.add_argument('--title', default=None, help='Optional chart title')
    p.add_argument('--year-min', type=int, default=None, help='Earliest year to include')
    p.add_argument('--year-max', type=int, default=None, help='Latest year to include')

    p.add_argument('--figsize', type=float, default=7.5)
    p.add_argument('--inner-radius', type=float, default=0.55)
    p.add_argument('--ring-width', type=float, default=0.28)
    p.add_argument('--ring-gap', type=float, default=0.02)

    p.add_argument('--cmap', default='coolwarm')
    p.add_argument('--symmetric', dest='symmetric', action='store_true')
    p.add_argument('--no-symmetric', dest='symmetric', action='store_false')
    p.set_defaults(symmetric=True)

    p.add_argument('--robust', dest='robust', action='store_true')
    p.add_argument('--no-robust', dest='robust', action='store_false')
    p.set_defaults(robust=True)

    p.add_argument('--quantiles', default='2,98')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--cbar-bins', type=int, default=10)

    p.add_argument('--year-axis-linewidth', type=float, default=0.9, help='Line width of the year axis line')
    p.add_argument('--year-tick-width', type=float, default=0.9, help='Line width of the year tick marks')
    p.add_argument('--show-year-labels', dest='show_year_labels', action='store_true')
    p.add_argument('--no-year-labels', dest='show_year_labels', action='store_false')
    p.set_defaults(show_year_labels=True)

    p.add_argument('--year-label-fontsize', type=int, default=5)
    p.add_argument('--year-label-step', type=int, default=None, help='Label/tick every Nth ring')
    p.add_argument('--max-year-labels', type=int, default=12)
    p.add_argument('--year-label-offset', type=float, default=0.0, help='Offset along axis-normal (axis units)')
    p.add_argument('--year-tick-length', type=float, default=0.04)

    p.add_argument('--center-label-fontsize', type=int, default=18)
    p.add_argument('--title-fontsize', type=int, default=12)
    p.add_argument('--cbar-tick-fontsize', type=int, default=9)
    p.add_argument('--cbar-label-fontsize', type=int, default=10)

    p.add_argument('--save', default=None)
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--cbar-fraction', type=float, default=0.045)
    p.add_argument('--cbar-shrink', type=float, default=1.0, help='Colorbar length multiplier (0<shrink<=1)')
    p.add_argument('--cbar-pad', type=float, default=0.03)
    p.add_argument('--cbar-segments', type=int, default=10)
    p.add_argument('--cbar-extend', default='both', choices=['neither','min','max','both'])

    p.add_argument('--month-labels', dest='add_month_labels', action='store_true')
    p.add_argument('--no-month-labels', dest='add_month_labels', action='store_false')
    p.set_defaults(add_month_labels=True)
    p.add_argument('--month-label-fontsize', type=int, default=10)
    p.add_argument('--month-label-offset', type=float, default=0.08)

    p.add_argument('--year-axis-angle-deg', type=float, default=180.0,
                   help='Angle of year axis (deg). Example: 165 places it between 9 and 10 o’clock.')
    p.add_argument('--cbar-step', type=float, default=None,
                   help='Exact colorbar/tick step (e.g., 0.1). Overrides cbar-segments/bins.')

    args = p.parse_args()

    # Load & coerce
    df = pd.read_csv(args.data)
    if args.year_col not in df.columns:
        raise ValueError(f"Year column '{args.year_col}' not found in CSV.")
    if args.month_col not in df.columns:
        raise ValueError(f"Month column '{args.month_col}' not found in CSV.")
    if args.value_col not in df.columns:
        raise ValueError(f"Value column '{args.value_col}' not found in CSV.")

    df[args.year_col] = pd.to_numeric(df[args.year_col], errors='coerce').astype('Int64')

    if pd.api.types.is_numeric_dtype(df[args.month_col]):
        df[args.month_col] = pd.to_numeric(df[args.month_col], errors='coerce').astype('Int64')
    else:
        df[args.month_col] = df[args.month_col].apply(lambda x: parse_month(x, args.month_format)).astype('Int64')

    df[args.value_col] = pd.to_numeric(df[args.value_col], errors='coerce')

    df = df.dropna(subset=[args.year_col, args.month_col])
    df = df[(df[args.month_col] >= 1) & (df[args.month_col] <= 12)]

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
        hide_center_label=args.hide_center_label or (args.center_label in (None, "", " ")),
        year_min=args.year_min,
        year_max=args.year_max,
        figsize=args.figsize,
        inner_radius=args.inner_radius,
        ring_width=args.ring_width,
        ring_gap=args.ring_gap,
        cmap=args.cmap,
        symmetric=args.symmetric,
        robust=True if args.robust else False,
        quantiles=(q_low, q_high),
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        show_year_labels=args.show_year_labels,
        cbar_bins=args.cbar_bins,
        year_label_fontsize=args.year_label_fontsize,
        year_label_step=args.year_label_step,
        max_year_labels=args.max_year_labels,
        year_tick_length=args.year_tick_length,
        cbar_fraction=args.cbar_fraction,
        cbar_shrink=args.cbar_shrink,
        cbar_pad=args.cbar_pad,
        cbar_segments=args.cbar_segments,
        cbar_extend=args.cbar_extend,
        dpi=args.dpi,
        save=args.save,
        center_label_fontsize=args.center_label_fontsize,
        title_fontsize=args.title_fontsize,
        cbar_tick_fontsize=args.cbar_tick_fontsize,
        cbar_label_fontsize=args.cbar_label_fontsize,
        year_label_offset=args.year_label_offset,
        add_month_labels=args.add_month_labels,
        month_label_fontsize=args.month_label_fontsize,
        month_label_offset=args.month_label_offset,
        year_axis_angle_deg=args.year_axis_angle_deg,
        year_axis_linewidth=args.year_axis_linewidth,
        year_tick_width=args.year_tick_width,
        cbar_step=args.cbar_step,
    )

if __name__ == '__main__':
    main()

