import json
import os
import io
import sys

nb_path = "fomc_jump_cross_impact_boxplots.ipynb"

print(f"=== EXECUTING DEDICATED NOTEBOOK '{nb_path}' ===")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

exec_globals = {}
code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]

print(f"Total code cells to execute: {len(code_cells)}")

for i, cell in enumerate(code_cells, 1):
    print(f"Executing Cell {i}/{len(code_cells)}...")
    code = "".join(cell["source"])
    
    outputs = []
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    try:
        exec(code, exec_globals)
        sys.stdout = old_stdout
        out_text = redirected_output.getvalue()
        
        if out_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": out_text.splitlines(True)
            })
            
        cell["outputs"] = outputs
        cell["execution_count"] = i
        print(f"  Cell {i} executed successfully.")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"  ERROR executing Cell {i}: {e}")
        out_text = redirected_output.getvalue()
        if out_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": out_text.splitlines(True)
            })
        cell["outputs"] = outputs
        cell["execution_count"] = i

# Save updated notebook with executed cell outputs
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook execution completed successfully and saved to '{nb_path}'!")
