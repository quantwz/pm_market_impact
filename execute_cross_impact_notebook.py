import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import duckdb
import io
import sys
from contextlib import redirect_stdout

import plotly.graph_objects as go
go.Figure.show = lambda *args, **kwargs: None
plt.show = lambda *args, **kwargs: None

nb_path = "cross_impact_news_shock_replay.ipynb"

print(f"=== EXECUTING DEDICATED NOTEBOOK '{nb_path}' ===")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution context
exec_globals = {
    '__name__': '__main__',
    'UNIVERSAL_DB': 'data/polymarket_orderbooks.db',
    'plt': plt,
    'go': go
}

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells to execute: {len(code_cells)}")

for idx, cell in enumerate(code_cells):
    code = "".join(cell['source'])
    if not code.strip():
        continue
    
    print(f"Executing Cell {idx + 1}/{len(code_cells)}...")
    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(code, exec_globals)
        
        output_text = stdout_capture.getvalue()
        cell_outputs = []
        if output_text:
            cell_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": output_text.splitlines(True)
            })
        
        cell['outputs'] = cell_outputs
        cell['execution_count'] = idx + 1
        print(f"  Cell {idx + 1} executed successfully.")
    except Exception as e:
        print(f"  ERROR executing Cell {idx + 1}: {e}")
        import traceback
        traceback.print_exc()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook execution completed successfully and saved to '{nb_path}'!")
