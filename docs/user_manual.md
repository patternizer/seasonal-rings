# User Manual — Radial Climatological Rings Chart

This manual describes the data requirements, plotting model, all CLI options, and troubleshooting tips for the **seasonality rings** chart.

---

## 1) Concept & Layout
Each **ring** is a year; rings proceed outward from the **most recent year** (inner) to the **earliest** (outer). Each ring has **12 wedges** for months, oriented like a clock:

- **December** at **12 o’clock** (top)
- **January** at ~1 o’clock
- ...
- **November** at ~11 o’clock

This orientation makes northern‑winter centered at the top.

The **year axis** is a radial guideline used for placing year labels and ticks. You can rotate it to any angle (degrees). By default it sits between **September** and **October** (`--year-axis-angle-deg 165`).

---

## 2) Data Requirements
- Input **CSV** with at least these columns:
  - `year` (integer)
  - `month` (1–12, or strings like `Jan`/`January`)
  - a numeric value column (e.g., `anomaly`, `average`)
- Use `--year-col`, `--month-col`, `--value-col` if your columns have different names.
- If the month is a string, the script will parse it automatically; you can force a style with `--month-format` = `auto|abbr|full|number`.
- Rows with invalid year/month are dropped; missing months are drawn in a neutral grey.

---

## 3) Color Scaling & Colorbar

### 3.1 Smart symmetric scaling
- **smart symmetric** (default **on**) inspects the data:
  - If the range **crosses zero** (e.g., anomalies), it uses a **symmetric** scale around 0.
  - If the range is **entirely positive/negative** (e.g., CO₂), it uses a **non‑symmetric** scale.
- Override behavior with `--symmetric` or `--no-symmetric`, or set explicit `--vmin/--vmax` for reproducible scales.

### 3.2 Robust min/max
- With `--robust` (default on), min/max are taken from **quantiles** (`--quantiles 2,98` by default) to reduce outlier impact.
- Use `--no-robust` to span the full min/max.

### 3.3 Steps, segments, and labels
- `--cbar-step` sets the **exact segment step** (e.g., `0.1`).
- If omitted, the script picks a **nice** step (1, 2, 2.5, 5 × 10^k) to achieve about `--cbar-segments` bins and **aligns** min/max to step boundaries.
- `--cbar-label-step` controls **label spacing** and is forced to be a multiple of `--cbar-step` to keep labels aligned to boundaries. If omitted, the script chooses a multiple to target `--cbar-bins` labels.
- Safety caps:
  - `--cbar-max-segments` (default 512) prevents drawing thousands of slivers.
  - `--cbar-max-ticks` (default 400) limits label count; extra labels are thinned.

### 3.4 Size & position
- `--cbar-fraction` controls **thickness**; `--cbar-shrink` controls **length**.
- `--cbar-pad` controls spacing from the main axes.
- `--cbar-extend` = `neither|min|max|both` adds triangular ends if values exceed plotted bounds.

---

## 4) Year Axis, Labels & Ticks
- **Angle**: `--year-axis-angle-deg` (degrees). 90° = 12 o’clock; angles **decrease clockwise**. Example: **165°** lands between **September** and **October**.
- **Label cadence**: `--year-label-step` (e.g., 10 → every 10th ring). If omitted, an auto step is chosen constrained by `--max-year-labels`.
- **Styling**:
  - `--year-label-fontsize`, `--year-label-weight`
  - `--year-label-offset` (offset along the axis-normal; increase to pull labels off the line)
  - `--year-axis-linewidth`, `--year-tick-width`, `--year-tick-length`
- Labels and ticks are **centered** on their corresponding ring midlines.

---

## 5) Month Labels
- Toggle with `--month-labels/--no-month-labels` (default on).
- Style with `--month-label-fontsize` and `--month-label-offset` (distance outside the outer ring).

---

## 6) Geometry & Appearance
- Ring order: **latest** year inner → older years outward.
- Radii: `--inner-radius`, `--ring-width`, `--ring-gap`.
- Figure size & DPI: `--figsize` (inches, square), `--dpi` (default 600).
- Colormap: any Matplotlib name via `--cmap` (diverging colormaps recommended for anomalies).
- Missing months use a neutral grey fill.
- Z‑order: wedges (bottom), center hole (below text), then axis/labels (top). Ensures labels are never hidden.

---

## 7) Full CLI Reference

### Required
- `--data PATH` — CSV file with monthly data
- `--value-col NAME` — numeric column to plot

### Data columns
- `--year-col NAME` (default: `year`)
- `--month-col NAME` (default: `month`)
- `--month-format {auto,abbr,full,number}` (default: `auto`)

### Time filtering
- `--year-min INT` — earliest year to include
- `--year-max INT` — latest year to include

### Figure & rings
- `--figsize FLOAT` (default 7.5)
- `--inner-radius FLOAT` (default 0.55)
- `--ring-width FLOAT` (default 0.28)
- `--ring-gap FLOAT` (default 0.02)

### Titles & labels
- `--title STR`
- `--center-label STR` (omit/empty to hide)
- `--hide-center-label` (force hide)
- `--title-fontsize INT` (default 12)
- `--center-label-fontsize INT` (default 18)

### Color scaling
- `--cmap STR` (default `coolwarm`)
- `--symmetric` / `--no-symmetric` (default symmetric; may be overridden by smart-symmetric)
- `--smart-symmetric` / `--no-smart-symmetric` (default on)
- `--robust` / `--no-robust` (default on)
- `--quantiles "L,H"` (default `2,98`)
- `--vmin FLOAT`, `--vmax FLOAT`

### Colorbar
- `--cbar-step FLOAT` — **exact** segment step; if omitted, auto-
- `--cbar-label-step FLOAT` — exact label spacing (multiple of step); if omitted, auto-
- `--cbar-segments INT` — target segments if step is auto (default 21)
- `--cbar-bins INT` — target label count for auto (default 10)
- `--cbar-max-segments INT` (default 512)
- `--cbar-max-ticks INT` (default 400)
- `--cbar-fraction FLOAT` — thickness (default 0.045)
- `--cbar-shrink FLOAT` — length multiplier (default 0.8)
- `--cbar-pad FLOAT` — gap from axes (default 0.03)
- `--cbar-extend {neither,min,max,both}` (default `both`)
- `--cbar-tick-fontsize INT` (default 9)
- `--cbar-label-fontsize INT` (default 10)

### Year axis, labels, ticks
- `--show-year-labels` / `--no-year-labels` (default on)
- `--year-axis-angle-deg FLOAT` (default 165)
- `--year-label-step INT` (default 10)
- `--max-year-labels INT` (default 9999)
- `--year-label-fontsize INT` (default 6)
- `--year-label-weight STR|INT` (default `bold`)
- `--year-label-offset FLOAT` (default 0.0)
- `--year-tick-length FLOAT` (default 0.08)
- `--year-axis-linewidth FLOAT` (default 1.4)
- `--year-tick-width FLOAT` (default 1.4)

### Month labels
- `--month-labels` / `--no-month-labels` (default on)
- `--month-label-fontsize INT` (default 10)
- `--month-label-offset FLOAT` (default 0.08)

### Output
- `--save PATH` — output filename (PNG). If omitted, a sensible name is used.
- `--dpi INT` — resolution (default 600)

---

## 8) Worked Examples

### 8.1 SST anomalies (symmetric, 0.1 steps)
```bash
python seasonal_ring_chart.py \
  --data HadSST.4.2.0.0_monthly_GLOBE.csv \
  --value-col anomaly --year-col year --month-col month \
  --year-min 1970 --year-max 2025 \
  --title "Global SST anomaly (HadSST4)" \
  --hide-center-label --month-labels \
  --year-axis-angle-deg 165 --year-label-step 10 \
  --cbar-step 0.1 --cbar-shrink 0.8 --cbar-fraction 0.045 --cbar-pad 0.03 \
  --year-label-fontsize 6 --year-label-weight bold --year-label-offset 0.22 \
  --year-axis-linewidth 1.4 --year-tick-width 1.4 --year-tick-length 0.10
```

### 8.2 Mauna Loa CO₂ (non-symmetric, auto steps)
```bash
python seasonal_ring_chart.py \
  --data co2_mm_mlo.csv \
  --value-col average --year-col year --month-col month \
  --year-min 1970 --year-max 2025 \
  --title "Mauna Loa CO₂ (ppm)" \
  --hide-center-label --month-labels \
  --year-axis-angle-deg 165 --year-label-step 10 \
  --cbar-shrink 0.6 --cbar-fraction 0.05 --cbar-pad 0.03 \
  --year-label-fontsize 8 --year-label-weight bold --year-label-offset 0.20 \
  --year-axis-linewidth 1.2 --year-tick-width 1.2 --year-tick-length 0.10
```

---

## 9) Troubleshooting

### "Locator attempting to generate N ticks … exceeds MAXTICKS"
- Cause: extremely fine `--cbar-step` over a very large range (e.g., 0.1 step across 800 units).
- Fix: omit `--cbar-step` (let auto choose), or increase it; alternatively set `--cbar-label-step` larger and keep `--cbar-step` small. You can also raise `--cbar-max-ticks` (default 400), but very high values clutter plots.

### Colorbar ticks don’t look aligned
- Ensure your tick spacing is a **multiple** of the segment step. If you set `--cbar-step`, the script forces `--cbar-label-step` to be a multiple; otherwise, set `--cbar-label-step` manually.

### Year labels hidden or mis-layered
- The plot enforces high z‑order for labels and axis elements. If labels still get busy, increase `--year-label-offset`, reduce `--year-label-fontsize`, or increase `--figsize`.

### Month names misinterpreted
- Use `--month-format abbr` for `Jan`/`Feb`/… or `--month-format full` for `January`/`February`/…

### Very long time ranges
- Consider smaller `--ring-width` and `--ring-gap`, a larger `--figsize`, and a bigger `--year-label-step` (e.g., 20 or 25).

---

## 10) Reproducibility & Publication Tips
- Fix `--vmin/--vmax` to maintain identical color scales across figures.
- Set `--dpi 600` or higher for print; `--dpi 300` is often fine for slides.
- Prefer a diverging colormap (e.g., `coolwarm`) for anomalies; sequential for absolute values (e.g., `viridis`).

---

## 11) Acknowledgements
Thanks for ideas and feedback that drove these features: robust scaling, aligned ticks/segments, label positioning, and axis rotation.

