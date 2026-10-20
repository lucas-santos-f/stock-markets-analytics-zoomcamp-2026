"""Homework 4: runs the course Module 4 notebook locally, once per question.

The [2026] Module 04 notebook is applied as is, with only the edits each question asks for:
  base  - unchanged, to check the original CAGR (1.1195)
  q1    - Ticker and month_wom dropped from CATEGORICAL before get_dummies
  q2    - q1 + tpr/fpr table on the random forest + rules for thresholds 0.56 to 0.64
  q3    - target redefined as growth_future_30d > 1.022

The neural network section is skipped (only precomputed history there, no predictions).
Needs pandas 2.x; the notebook breaks on pandas 3.

python sma_hw4.py <module04.ipynb> <stocks_df_combined_2026_09_18.parquet.brotli> q1
"""
import sys

import nbformat
from nbclient import NotebookClient

src_path, parquet, variant = sys.argv[1:4]

nb = nbformat.read(src_path, as_version=4)
cell = nbformat.v4.new_code_cell

cells = [cell("import matplotlib\nmatplotlib.use('Agg')")]
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code' or 138 <= i <= 155:
        continue
    src = '\n'.join(l for l in c.source.split('\n')
                    if not l.strip().startswith(('!', '%%time')))
    src = src.replace('.sample(10)', '.head(10)')
    src = src.replace('/content/stocks_df_combined_2026_09_18.parquet.brotli', parquet.replace('\\', '/'))
    src = src.replace('fig.show()', 'pass')

    if variant == 'q3' and i == 8:
        src += "\ndf_full['is_positive_growth_30d_future'] = (df_full['growth_future_30d'] > 1.022).astype(int)"
    if variant in ('q1', 'q2') and i == 38:
        src = "CATEGORICAL = [c for c in CATEGORICAL if c not in ('Ticker', 'month_wom')]\n" + src
    if variant in ('q1', 'q2') and i == 43:
        src += "\nprint('features:', len(NUMERICAL + DUMMIES))"
    if variant == 'q2' and i == 158:
        src = src.replace('y_pred_test = clf_best', '# y_pred_test = clf_best')
        src = src.replace('# y_pred_test = rf_best', 'y_pred_test = rf_best')
    if variant == 'q2' and i == 176:
        src += '\n' + '\n'.join(
            f"new_df['pred{12 + k}_rf_best_rule_{56 + k}'] = (y_pred_all_class1_array >= 0.{56 + k}).astype(int)"
            for k in range(9))
    cells.append(cell(src))

cells.append(cell(
    "print(df_sim1_results[['prediction', 'sim1_count_investments', 'sim1_avg_investments_per_day', "
    "'sim1_capital', 'sim1_CAGR']].to_string())"))
nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
NotebookClient(nb, timeout=3600, kernel_name='python3').execute()

for c in nb.cells[-1:] + [c for c in nb.cells if 'features:' in c.source]:
    for o in c.get('outputs', []):
        print(o.get('text', ''))
