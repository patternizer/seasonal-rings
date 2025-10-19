#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: seasonal_rings.py
#------------------------------------------------------------------------------
# Version 1.0
# 19 October, 2025
# michael.taylor@cefas.gov.uk
# https://patternizer.github.io
#------------------------------------------------------------------------------
"""
Generic radial climatological ring chart for monthly data.

Key features:
- December at 12 o'clock; January at 1 o'clock; ... November at 11 o'clock.
- Inner ring = latest year; outer = earliest.
- Smart symmetric scaling (auto-switch to non-symmetric if data doesn't cross zero).
- Automatic "nice" colorbar step and label spacing unless explicitly set.
- Caps for colorbar segments/ticks to avoid Matplotlib overload.
- Month labels around the outside; configurable year axis angle; bold year labels supported.
- Headless rendering (Agg); always saves PNG and closes the figure.

Angle convention: 12 o'clock = 90°, 3 o'clock = 0°, angles decrease clockwise.
"""
#------------------------------------------------------------------------------


# ----------------------
# IMPORT LIBRARIES
# ----------------------

import argparse
import math
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")  # ensure no GUI
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

# ---- "nice" step utilities ----

def _nice_step(data_range: float, target_steps: int) -> float:
    """Return a nice step (1, 2, 2.5, 5 × 10^k) close to data_range/target_steps."""
    if data_range <= 0 or target_steps <= 0:
        return 1.0
    raw = data_range / target_steps
    exp = math.floor(math.log10(raw))
    base = raw / (10 ** exp)
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if base <= m:
            return m * (10 ** exp)
    return 10.0 * (10 ** exp)

def _align_to_step(vmin: float, vmax: float, step: float, symmetric: bool) -> Tuple[float, float]:
    """Round vmin/vmax to multiples of step for clean boundaries; symmetric keeps ±bound."""
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        bound = math.ceil(bound / step) * step
        return -bound, bound
    vmin_a = math.floor(vmin / step) * step
    vmax_a = math.ceil(vmax / step) * step
    return vmin_a, vmax_a

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
    symmetric: Optional[bool] = True,
    smart_symmetric: bool = True,
    robust: bool = True,
    quantiles: Tuple[float, float] = (2, 98),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    show_year_labels: bool = True,
    year_label_fontweight: str = 'bold',
    cbar_bins: int = 10,
    cbar_segments: int = 21,
    cbar_extend: str = 'both',
    year_label_fontsize: int = 6,
    year_label_step: Optional[int] = 10,
    max_year_labels: int = 9999,
    year_tick_length: float = 0.08,
    cbar_fraction: float = 0.045,   # thickness
    cbar_shrink: float = 0.8,       # length
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
    year_axis_angle_deg: float = 165.0,
    # colorbar controls
    cbar_step: Optional[float] = None,     # if None, auto
    cbar_label_step: Optional[float] = None,  # if None, auto (multiple of cbar_step)
    cbar_max_ticks: int = 400,
    cbar_max_segments: int = 512,
    dpi: int = 600,
    save: Optional[str] = None,
):
    # Filter year range
    years_all = sorted(df[year_col].dropna().astype(int).unique())
    if not years_all:
        raise ValueError("No valid years in the data.")
    if year_min is None:
        year_min = min(years_all)
    if year_max is None:
        year_max = max(years_all)
    years = [y for y in years_all if year_min <= y <= year_max]
    if not years:
        raise ValueError("No data within the specified year range.")
    years_sorted = sorted(years, reverse=True)
    n_rings = len(years_sorted)

    # Determine year label step (explicit beats auto)
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

    # Value map and data range
    val_map = build_value_map(df, year_col, month_col, value_col)
    vals = np.array([float(v) for v in val_map.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        vals = np.array([0.0])

    # Determine symmetric vs non-symmetric smartly if requested
    symmetric_use = bool(symmetric)
    if smart_symmetric and vmin is None and vmax is None:
        vmin_raw, vmax_raw = float(np.nanmin(vals)), float(np.nanmax(vals))
        crosses_zero = (vmin_raw < 0) and (vmax_raw > 0)
        if not crosses_zero:
            symmetric_use = False

    # vmin/vmax selection (robust or full)
    if (vmin is None) or (vmax is None):
        low, high = (np.nanpercentile(vals, quantiles) if robust
                     else (float(np.nanmin(vals)), float(np.nanmax(vals))))
        if symmetric_use:
            bound = max(abs(low), abs(high))
            vmin_use, vmax_use = -bound, bound
        else:
            vmin_use, vmax_use = low, high
    else:
        vmin_use, vmax_use = vmin, vmax

    # Auto step for colorbar segments if not provided
    data_range = vmax_use - vmin_use if (vmax_use is not None and vmin_use is not None) else 1.0
    if cbar_step is None or cbar_step <= 0:
        # aim for ~cbar_segments bins
        step_auto = _nice_step(data_range, max(2, int(cbar_segments)))
        cbar_step_use = step_auto
    else:
        cbar_step_use = float(cbar_step)

    # Align min/max to step boundaries
    vmin_use, vmax_use = _align_to_step(vmin_use, vmax_use, cbar_step_use, symmetric_use)

    # Build levels at the chosen step, with a segment cap
    levels = np.arange(vmin_use, vmax_use + 0.5 * cbar_step_use, cbar_step_use)
    if (len(levels) - 1) > cbar_max_segments:
        # coarsen to respect max segments
        mult = math.ceil((len(levels) - 1) / cbar_max_segments)
        cbar_step_use *= mult
        vmin_use, vmax_use = _align_to_step(vmin_use, vmax_use, cbar_step_use, symmetric_use)
        levels = np.arange(vmin_use, vmax_use + 0.5 * cbar_step_use, cbar_step_use)

    # Colormap / norm
    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
    cmap_obj = ListedColormap(colors)
    try:
        cmap_obj.set_bad(missing_color)
    except Exception:
        pass
    norm = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=False)
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

    # Rings (lowest zorder)
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

        # Year labels (on top)
        if show_year_labels and (step is not None) and (i % step == 0):
            r_mid = 0.5 * (r_in + r_out)
            theta = np.deg2rad(year_axis_angle_deg)
            nx, ny = -np.sin(theta), np.cos(theta)  # axis-normal
            x = r_mid * np.cos(theta) + year_label_offset * nx
            y = r_mid * np.sin(theta) + year_label_offset * ny
            ax.text(x, y, str(year), va='center', ha='center',
                    fontsize=year_label_fontsize, fontweight=year_label_fontweight,
                    zorder=200, clip_on=False)

    # Central hole below text
    hole = Circle((0, 0), radius=inner_radius - 0.01, facecolor='white', edgecolor='none', zorder=2)
    ax.add_patch(hole)
    if (not hide_center_label) and center_label:
        ax.text(0, 0, center_label, ha='center', va='center',
                fontsize=center_label_fontsize, fontweight='bold', zorder=250)

    # Year axis line (top)
    theta = np.deg2rad(year_axis_angle_deg)
    x1, y1 = (inner_radius + ring_width) * np.cos(theta), (inner_radius + ring_width) * np.sin(theta)
    x2, y2 = R_max * np.cos(theta), R_max * np.sin(theta)
    # widths provided via closure; default set later in main() pass-through
    ax.plot([x1, x2], [y1, y2], color='black',
            linewidth=year_axis_linewidth, zorder=150)

    # Tick marks along axis for labeled rings (top)
    if show_year_labels:
        for i, year in enumerate(years_sorted):
            label_this = (i % (year_label_step if year_label_step else step) == 0)
            if label_this:
                r_in_i = inner_radius + i * (ring_width + ring_gap)
                r_out_i = r_in_i + ring_width
                r_mid_i = 0.5 * (r_in_i + r_out_i)
                px = r_mid_i * np.cos(theta)
                py = r_mid_i * np.sin(theta)
                nx, ny = -np.sin(theta), np.cos(theta)
                dx, dy = (year_tick_length/2) * nx, (year_tick_length/2) * ny
                ax.plot([px - dx, px + dx], [py - dy, py + dy],
                        color='black', linewidth=year_tick_width, zorder=150)

    # Month labels (top)
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

    # Colorbar: ticks aligned to boundary multiples, with caps
    sm.set_array([])
    # choose label step (multiple of cbar_step_use)
    if cbar_label_step is None or cbar_label_step <= 0:
        # aim for ~cbar_bins labels, but keep as multiple of step
        approx = _nice_step(vmax_use - vmin_use, max(2, cbar_bins))
        mult = max(1, int(round(approx / cbar_step_use)))
        cbar_label_step_use = mult * cbar_step_use
    else:
        # enforce integer multiple for alignment
        mult = max(1, int(round(cbar_label_step / cbar_step_use)))
        cbar_label_step_use = mult * cbar_step_use

    start = math.ceil(vmin_use / cbar_label_step_use) * cbar_label_step_use
    ticks = np.arange(start, vmax_use + 0.5 * cbar_label_step_use, cbar_label_step_use)
    if ticks.size > cbar_max_ticks:
        mthin = int(math.ceil(ticks.size / cbar_max_ticks))
        ticks = ticks[::mthin]

    cbar = fig.colorbar(sm, ax=ax,
                        fraction=cbar_fraction, pad=cbar_pad,
                        boundaries=levels, spacing='proportional',
                        extend=cbar_extend, ticks=ticks,
                        drawedges=True, shrink=cbar_shrink)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar.ax.set_ylabel(value_col, rotation=270, va='bottom', fontsize=cbar_label_fontsize)

    # Set sensible decimals based on label step
    step = cbar_label_step_use
    if abs(step - round(step)) < 1e-9:
        fmt = '%.0f'
    elif abs(step*10 - round(step*10)) < 1e-9:
        fmt = '%.1f'
    elif abs(step*100 - round(step*100)) < 1e-9:
        fmt = '%.2f'
    else:
        fmt = '%.3f'
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

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
    p = argparse.ArgumentParser(description="Concentric radial rings chart for monthly data (December at 12 o'clock).")
    p.add_argument('--data', required=True, help='Path to CSV file with monthly data')
    p.add_argument('--value-col', required=True, help='Column with numeric monthly values')
    p.add_argument('--year-col', default='year', help='Year column (integer)')
    p.add_argument('--month-col', default='month', help='Month (1..12 or names)')
    p.add_argument('--month-format', default='auto', choices=['auto', 'abbr', 'full', 'number'])

    p.add_argument('--center-label', default=None, help='Label inside central hole (omit/empty to hide)')
    p.add_argument('--hide-center-label', action='store_true', help='Do not draw label in the center')
    p.add_argument('--title', default=None, help='Optional chart title')
    p.add_argument('--year-min', type=int, default=1970)
    p.add_argument('--year-max', type=int, default=2025)

    p.add_argument('--figsize', type=float, default=7.5)
    p.add_argument('--inner-radius', type=float, default=0.55)
    p.add_argument('--ring-width', type=float, default=0.28)
    p.add_argument('--ring-gap', type=float, default=0.02)

    p.add_argument('--cmap', default='coolwarm')
    p.add_argument('--symmetric', dest='symmetric', action='store_true', help='Force symmetric scaling')
    p.add_argument('--no-symmetric', dest='symmetric', action='store_false', help='Force non-symmetric scaling')
    p.set_defaults(symmetric=True)
    p.add_argument('--smart-symmetric', dest='smart_symmetric', action='store_true',
                   help='Auto switch to non-symmetric if data does not cross zero (default on)')
    p.add_argument('--no-smart-symmetric', dest='smart_symmetric', action='store_false')
    p.set_defaults(smart_symmetric=True)

    p.add_argument('--robust', dest='robust', action='store_true')
    p.add_argument('--no-robust', dest='robust', action='store_false')
    p.set_defaults(robust=True)

    p.add_argument('--quantiles', default='2,98')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)

    p.add_argument('--cbar-bins', type=int, default=20, help='Target count for label placement (used by auto)')
    p.add_argument('--cbar-segments', type=int, default=41, help='Target segments if step is auto')
    p.add_argument('--cbar-step', type=float, default=None, help='Exact step for color segments (auto if omitted)')
    p.add_argument('--cbar-label-step', type=float, default=None, help='Exact label spacing (multiple of step; auto if omitted)')
    p.add_argument('--cbar-max-ticks', type=int, default=50, help='Max colorbar tick labels')
    p.add_argument('--cbar-max-segments', type=int, default=512, help='Max discrete segments to draw')
    p.add_argument('--cbar-extend', default='both', choices=['neither','min','max','both'])
    p.add_argument('--cbar-fraction', type=float, default=0.05)
    p.add_argument('--cbar-shrink', type=float, default=0.75, help='Colorbar length multiplier (0<shrink<=1)')
    p.add_argument('--cbar-pad', type=float, default=0.03)
    p.add_argument('--cbar-tick-fontsize', type=int, default=10)
    p.add_argument('--cbar-label-fontsize', type=int, default=10)

    p.add_argument('--show-year-labels', dest='show_year_labels', action='store_true')
    p.add_argument('--no-year-labels', dest='show_year_labels', action='store_false')
    p.set_defaults(show_year_labels=True)
    p.add_argument('--year-label-fontsize', type=int, default=6)
    p.add_argument('--year-label-weight', default='normal', help="Font weight for year labels (e.g. 'normal', 'bold', 600)")
    p.add_argument('--year-label-step', type=int, default=10, help='Label/tick every Nth ring')
    p.add_argument('--max-year-labels', type=int, default=100)
    p.add_argument('--year-label-offset', type=float, default=1.0, help='Offset along axis-normal (axis units)')

    p.add_argument('--year-tick-length', type=float, default=0.5)
    p.add_argument('--year-axis-linewidth', type=float, default=1.0)
    p.add_argument('--year-tick-width', type=float, default=0.5)
    p.add_argument('--year-axis-angle-deg', type=float, default=165.0,
                   help='Angle of year axis (deg). 165° is between Sep & Oct.')

    p.add_argument('--month-labels', dest='add_month_labels', action='store_true')
    p.add_argument('--no-month-labels', dest='add_month_labels', action='store_false')
    p.set_defaults(add_month_labels=True)
    p.add_argument('--month-label-fontsize', type=int, default=10)
    p.add_argument('--month-label-offset', type=float, default=0.08)

    p.add_argument('--center-label-fontsize', type=int, default=18)
    p.add_argument('--title-fontsize', type=int, default=12)

    p.add_argument('--save', default=None)
    p.add_argument('--dpi', type=int, default=600)

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

    # Expose two linewidth params via closure
    global year_axis_linewidth, year_tick_width
    year_axis_linewidth = args.year_axis_linewidth
    year_tick_width = args.year_tick_width

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
        smart_symmetric=args.smart_symmetric,
        robust=True if args.robust else False,
        quantiles=(q_low, q_high),
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        show_year_labels=args.show_year_labels,
        year_label_fontweight=args.year_label_weight,
        cbar_bins=args.cbar_bins,
        cbar_segments=args.cbar_segments,
        cbar_extend=args.cbar_extend,
        year_label_fontsize=args.year_label_fontsize,
        year_label_step=args.year_label_step,
        max_year_labels=args.max_year_labels,
        year_tick_length=args.year_tick_length,
        cbar_fraction=args.cbar_fraction,
        cbar_shrink=args.cbar_shrink,
        cbar_pad=args.cbar_pad,
        center_label_fontsize=args.center_label_fontsize,
        title_fontsize=args.title_fontsize,
        cbar_tick_fontsize=args.cbar_tick_fontsize,
        cbar_label_fontsize=args.cbar_label_fontsize,
        year_label_offset=args.year_label_offset,
        add_month_labels=args.add_month_labels,
        month_label_fontsize=args.month_label_fontsize,
        month_label_offset=args.month_label_offset,
        missing_color='#d9d9d9',
        year_axis_angle_deg=args.year_axis_angle_deg,
        cbar_step=args.cbar_step,
        cbar_label_step=args.cbar_label_step,
        cbar_max_ticks=args.cbar_max_ticks,
        cbar_max_segments=args.cbar_max_segments,
        dpi=args.dpi,
        save=args.save,
    )

if __name__ == '__main__':
    main()

