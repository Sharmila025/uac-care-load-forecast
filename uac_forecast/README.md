# UAC Care Load & Placement Demand Forecasting

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The CSV is already in `data/`. You can also upload a replacement in the sidebar.

## Files
- `engine.py`: cleaning, daily series, features, 6 models, walk-forward validation, KPIs
- `app.py`: Streamlit dashboard (red/black theme)
- `.streamlit/config.toml`: theme

## Notes
- First load runs the walk-forward validation (about 1 minute), then it is cached.
- Tick "Include SARIMA" in the sidebar to add SARIMA (needs statsmodels).
- The dataset has no official capacity number, so the stress threshold is user-set.
