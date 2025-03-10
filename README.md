# Citation Network Analysis

A Python-based tool for analyzing academic citation networks between institutions, focusing on comparing research impact between different institutions.

## Features

- Fetches academic publication data from Google Scholar
- Builds and analyzes citation networks between institutions
- Generates comprehensive network analysis metrics
- Creates interactive visualizations
- Produces detailed analysis reports

## Requirements

```
networkx
matplotlib
pandas
numpy
seaborn
scholarly
requests
```

## Project Structure

- `app.py` - Main application with core functionality
- `report_generator.py` - Generates comprehensive analysis reports
- `utils.py` - Utility functions for file operations and plot saving
- `visualize_metrics.py` - Visualization functions for different metrics

## Usage

1. Install dependencies:
```bash
pip install networkx matplotlib pandas numpy seaborn scholarly requests
```

2. Run the analysis:
```python
from app import run_full_analysis

# Run analysis for two institutions
G, network_data, analysis = run_full_analysis(
    iit_name="IIT Madras",
    nit_name="NIT Trichy",
    use_cache=True  # Set to False to fetch fresh data
)
```

## Analysis Metrics

The tool analyzes various network metrics including:
- PageRank
- Eigenvector Centrality
- Betweenness Centrality
- Closeness Centrality
- Hub and Authority Scores
- Cross-institution citations
- Institution-wise citation patterns

## Visualizations

Generated visualizations include:
1. Citation Network Graph
2. Institution Citation Matrix
3. Professor Citation Analysis
4. Citation Trends
5. Centrality Distributions
6. Various Metric-specific Visualizations

## Output

The analysis generates:
- Interactive network visualizations
- Comprehensive analysis report in text format
- Publication and citation statistics
- Cross-institution collaboration metrics
- Individual professor performance metrics

## File Structure

```
Project/
├── app.py              # Main application
├── report_generator.py # Report generation
├── utils.py           # Utility functions
├── visualize_metrics.py# Visualization code
├── plots/             # Generated visualizations
├── outputs/           # Analysis reports
└── README.md          # Documentation
```

## Note

Due to Google Scholar's rate limiting, the tool includes caching functionality. Set `use_cache=False` in `run_full_analysis()` to fetch fresh data.
