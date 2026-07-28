"""Swap the changed cells into the three already-executed three_blocks notebooks.

`_make_three_block_notebooks.py` rebuilds a notebook from scratch, which throws away every cell's
stored output. Only SS 9a changed substantively (LDA + KMeans panels, recall matrices below, N swept
1, 5, 10, ... , 150) - plus Reviewer #3, which shares `cross_agg_curve` and `NS_CROSS` with it. This
patches those cells and leaves the rest - and all their existing outputs - untouched. The patched
cells are cleared, so re-running them is required.
"""
import importlib.util
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["playstyle_v2_raw_v1to4.ipynb", "playstyle_v2_choice.ipynb", "playstyle_v2_both.ipynb"]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "_gen", os.path.join(HERE, "_make_three_block_notebooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    gen = _generator()
    # (cell_type, source prefix that identifies the cell) -> replacement source
    edits = [("code",     "# ===== SS 9a",      gen.C_CROSS_A.strip("\n")),
             ("code",     "# ===== Reviewer #3", gen.C_REV3.strip("\n")),
             ("markdown", "## 9 - Cross-deck",   gen.MD_CROSS)]
    for fname in TARGETS:
        path = os.path.join(HERE, fname)
        nb = nbf.read(path, as_version=4)
        hits = {p: 0 for _, p, _ in edits}
        for cell in nb.cells:
            for ctype, prefix, new_src in edits:
                if cell.cell_type == ctype and cell.source.lstrip().startswith(prefix):
                    cell.source = new_src
                    hits[prefix] += 1
                    if ctype == "code":     # stale: the code that produced them is gone
                        cell.outputs = []
                        cell.execution_count = None
        if set(hits.values()) != {1}:
            raise SystemExit(f"{fname}: expected exactly one cell per prefix, got {hits} - not writing")
        nbf.write(nb, path)
        kept = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
        print(f"patched {fname}  ({', '.join(hits)} replaced; {kept} outputs kept elsewhere)")


if __name__ == "__main__":
    main()
