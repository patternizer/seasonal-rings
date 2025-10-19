# seasonal-rings — Radial Climatological Rings Chart (Python)

Concentric seasonal rings for monthly data. Each ring is a year (inner = most recent), and each ring has 12 wedges (Dec at 12 o’clock, Jan at 1 o’clock, …, Nov at 11 o’clock). Designed for anomalies **and** absolute values (e.g., CO₂), with smart color scaling, publication-ready output, and a friendly CLI.

- 📄 **User Manual:** see [`docs/User-Manual.md`](docs/User-Manual.md)

---

## ✨ Features

- **Rings by year**: inner = latest, outer = earliest.  
- **Clockwise month layout**: **Dec at 12 o’clock** (top), then Jan ≈ 1 o’clock, …, Nov ≈ 11 o’clock.  
- **Smart color scaling**:
  - *Smart symmetric* (default): auto-detects if data crosses 0; uses symmetric scale for anomalies, non-symmetric otherwise.
  - Optional robust min/max via quantiles (`--quantiles`).
  - Exact colorbar step control (e.g., **`--cbar-step 0.1`**), with automatic **tick thinning** to avoid over-plotting.
- **Aligned colorbar**: tick labels and step boundaries line up.
- **Month labels** around the outside (toggle on/off).
- **Configurable year axis**: angle, tick spacing, label offset, bold font, widths.
- **Headless**: saves straight to PNG; no GUI window.

---

## 📦 Install

You can use **Conda** (recommended) or fall back to a local **venv**. The repo includes cross-platform setup scripts.

### Option A — Linux/macOS (Conda or venv fallback)

```bash
chmod +x setup.sh
./setup.sh

# If Conda/Mamba was found:
conda activate seasonal-rings
# Otherwise a local venv was created:
source .venv/bin/activate
```

### Option B — Windows (Conda or venv fallback)

```bat
setup.bat

:: If Conda/Mamba was found:
conda activate seasonal-rings
:: Otherwise a local venv was created:
.\.venv\Scripts ctivate
```

> The environment spec lives in [`environment.yml`](environment.yml). The setup scripts will use Conda/Mamba if available; otherwise they create `.venv` and install `numpy`, `pandas`, and `matplotlib` via `pip`.

---

## 🚀 Quick start


```bash
python seasonal_rings.py --data <file.csv> --value-col <name> --year-col <name> --month-col <name> [options]
```

## 🧭 Custom usage (CLI)

### Example A — Anomalies (symmetric, 0.1 steps)

```bash
python seasonal_rings.py   --data examples/HadSST.4.2.0.0_monthly_GLOBE.csv   --value-col anomaly --year-col year --month-col month   --year-min 1970 --year-max 2025   --title "Global SST anomaly (HadSST4)"   --hide-center-label --month-labels   --year-axis-angle-deg 165 --year-label-step 10   --cbar-step 0.1 --cbar-shrink 0.8 --cbar-fraction 0.045 --cbar-pad 0.03   --year-label-fontsize 6 --year-label-weight bold --year-label-offset 0.22   --year-axis-linewidth 1.4 --year-tick-width 1.4 --year-tick-length 0.10
```

### Example B — Positive-only values (e.g., CO₂ ppm; non-symmetric auto)

```bash
python seasonal_rings.py   --data examples/co2_mm_mlo.csv   --value-col average --year-col year --month-col month   --year-min 1970 --year-max 2025   --title "Mauna Loa CO₂ (ppm)"   --hide-center-label --month-labels   --year-axis-angle-deg 165 --year-label-step 10   --cbar-shrink 0.6 --cbar-fraction 0.05 --cbar-pad 0.03   --year-label-fontsize 8 --year-label-weight bold --year-label-offset 0.20   --year-axis-linewidth 1.2 --year-tick-width 1.2 --year-tick-length 0.10
```

> If you omit `--cbar-step`, the script picks a **nice** step and aligns min/max to step boundaries while capping segments/ticks for readability.

---

**Data expectations**

- CSV with **year**, **month**, and a numeric **value** column.  
- Month can be **1–12** or strings (`Jan`, `January`). Use `--month-format` if needed.  
- Missing months are drawn in a neutral grey.

**Month layout**

- Rotated so **December sits at 12 o’clock** (top). January ≈ 1 o’clock, etc.

> For the **full option reference**, scaling logic, troubleshooting, and more examples, see the **[User Manual](docs/User-Manual.md)**.

---

## 🖼 Output

- Always saves a PNG (default name: `seasonal_ring_chart_<value>_<firstYear>-<lastYear>.png`).  
- Use `--save out.png` to set the filename.  
- Default `--dpi 600` (publication quality).

---

## 🗂 Repo layout

```
.
├── seasonal_rings.py          # main script
├── environment.yml            # Conda env spec
├── setup.sh                   # Linux/macOS setup (Conda or venv fallback)
├── setup.bat                  # Windows setup (Conda or venv fallback)
├── examples/                  # sample data CSVs
├── docs/
│   └── User-Manual.md         # full manual
├── images/                    # output PNG images
└── LICENSE.md
```

---

## 🧪 Development notes

- Headless backend (**Agg**); no windows pop up.  
- Z-order is set so data wedges are bottom; labels/axis are on top.  
- Performance is O(12 × number-of-years) and scales well.

---

## 🤝 Contributing

Issues and PRs are welcome. When filing an issue, please include:

- a minimal CSV sample (or a snippet),
- the exact command you ran,
- the observed vs. expected output (and a PNG if possible).

 Contact information:

* [Michael Taylor](michael.taylor@cefas.gov.uk)

---

## 📄 License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

See [`LICENSE.md`](LICENSE.md).
