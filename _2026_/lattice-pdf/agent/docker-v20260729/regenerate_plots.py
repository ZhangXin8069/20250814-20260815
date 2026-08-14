import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ratio import run_analysis
from utils import setup_logging

config = json.load(open('run_config.json'))
latest = sorted(Path('.').glob('output_*'))[-1]
logger = setup_logging(latest / 'run.log', 'plot_regenerate')
results = run_analysis(config, latest / 'data', latest / 'plots', logger)
print(f'Status: {results["status"]}')
print(f'Plots:')
for k in ['ratio_path','diag_path','meff_path','field_strength_path']:
    print(f'  {results.get(k)}')
