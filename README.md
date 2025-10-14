# LTV Reach Calculation

A comprehensive LTV (Linear TV) reach calculation pipeline that computes daily cumulative reach using sophisticated progressive growth logic.

## Overview

This project implements the LTV reach calculation logic inherited from the cross_media_dashboard project, providing accurate TV campaign reach analysis with enhanced low GRP campaign logic.

## Features

- **Spot Data Processing**: Fetches TV spot data from ClickHouse database
- **Channel Reach Calculation**: Computes channel-specific reach using share data
- **Daily Cumulative Reach**: Applies progressive growth logic for accurate reach modeling
- **LTV Reach Analysis**: Final reach calculation with campaign type classification
- **Tabular Output**: Clean, formatted results for easy analysis

## Pipeline Steps

1. **Spot Data Collection**: Fetch TV spot data from ClickHouse
2. **Channel Reach Calculation**: Compute channel-specific reach using share data
3. **Daily Cumulative Reach**: Apply progressive growth logic (Low GRP vs Standard)
4. **LTV Reach Analysis**: Extract final reach values for each region/brand

## Enhanced Low GRP Campaign Logic

For campaigns with **max_reach < 40%**, the system uses Enhanced Progressive Growth Logic:

- **Baseline**: At rating = 58.4, reach = max_reach
- **1st Increment**: At rating = 116.8, reach = max_reach × 1.04 (4% growth)
- **2nd Increment**: At rating = 175.2, reach = (max_reach × 1.04) × 1.08 (8% growth)
- **3rd Increment**: At rating = 233.6, reach = ((max_reach × 1.04) × 1.08) × 1.12 (12% growth)
- **And so on...** with increasing growth rates at each 58.4 rating interval

## Usage

### Streamlit Dashboard (Recommended)

```bash
# Run the interactive dashboard
streamlit run app.py
```

The dashboard provides:
- Interactive filters for product, date range, and regions
- Real-time LTV reach calculation
- Visual table display with search and pagination
- CSV export functionality
- Summary metrics and statistics

### Command Line

```bash
# Run the LTV reach calculation pipeline
python modules/main.py
```

## Configuration

Update the parameters in `modules/main.py`:

```python
# Example parameters - modify as needed
products = "vim drop"  # Change to your product name
start_date = "2025-01-01"  # Change to your start date
end_date = "2025-01-31"    # Change to your end date
regions = ['Delhi', 'TN/Pondicherry']  # Change to your regions or None for all
```

## Output

The pipeline generates:

- **Daily LTV Reach Table**: Complete daily progression with all metrics
- **LTV Reach Summary**: Key statistics and final reach values
- **CSV Files**: All results saved in the `output/` folder

## Project Structure

```
├── modules/
│   ├── main.py              # Main pipeline execution
│   ├── config.py            # Database and configuration
│   ├── get_data.py          # Spot data retrieval
│   ├── channel_reach.py     # Channel reach computation
│   ├── daily_cum_reach.py   # Daily cumulative reach logic
│   ├── utils.py             # Data formatting utilities
│   └── core/
│       ├── __init__.py
│       └── ltv_reach.py     # LTV reach business logic
├── inputs/                  # Input data files
└── output/                  # Generated results
```

## Requirements

- Python 3.8+
- pandas
- numpy
- requests
- streamlit (for error handling)

## Database Configuration

The system connects to ClickHouse database for spot data and channel share information. Update the database configuration in `modules/config.py` as needed.

## License

This project inherits the logic from the cross_media_dashboard project and maintains compatibility with the original LTV analyzer functionality.
