![image](https://github.com/patternizer/seasonal-rings/blob/main/images/seasonal_ring_chart_anomaly_1970-2025.png)

# seasonal-rings

Concentric “seasonal ring” chart for monthly data.  
Each ring is a year, with **December at 12 o’clock**, **January at 1 o’clock**, … **November at 11 o’clock**.  
The center is a donut hole labelled **SST** (configurable), and colors show the monthly values (anomalies by default).

![example](docs/example.png) <!-- optional image; remove if not used -->

## Features
- Concentric rings: inner = latest year (e.g., 2025 if present), then 2024, 2023, … outward to earliest (or a user-set range).
- 12 aligned slices per ring (clock layout).
- Central label (default **SST**).
- Sensible colorbar (~10 ticks); symmetric color scale around 0 by default (good for anomalies).
- Handles missing months (neutral grey).

## Quick start

### Create the environment

$ chmod +x setup.sh
$ ./setup.sh

### Run

$ python seasonal_ring_chart.py --data examples\\co2_mm_mlo.csv --value-col average
$ python seasonal_ring_chart.py --data examples\\HadSST.4.2.0.0_monthly_GLOBE.csv --value-col anomaly
$ python seasonal_ring_chart.py --data examples\\HadSST.4.2.0.0_monthly_NHEM.csv --value-col anomaly

(custom)

$ python seasonal_ring_chart.py \
  --data examples\HadSST.4.2.0.0_monthly_NHEM.csv \
  --value-col anomaly --year-col year --month-col month \
  --year-min 1970 --year-max 2025 \
  --title "NH SST anomaly (HadSST4)" \
  --hide-center-label --month-labels \
  --year-axis-angle-deg 165 --year-label-step 10 \
  --cbar-shrink 0.75 --cbar-fraction 0.05 --cbar-pad 0.03 \
  --year-label-fontsize 6 --year-label-weight normal --year-label-offset 1 \
  --year-axis-linewidth 1 --year-tick-width 0.5 --year-tick-length 0.5

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.taylor@cefas.gov.uk)
