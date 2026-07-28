"""Refresh named cells in the already-executed three_blocks notebooks, keeping every other output.

`_make_three_block_notebooks.py` rebuilds a notebook from scratch, which throws away every stored
output - expensive here, since SS 8 and SS 9a take minutes each. This builds a fresh notebook in
memory and copies the source of the cells listed in `PREFIXES` onto the matching cells on disk.
Patched code cells are cleared, so re-running them is required; every other cell keeps its output.

Set PREFIXES to whatever changed, then run. (Supersedes `_patch_ss9a.py`, which hard-coded the
SS 9a / Reviewer #3 / MD_CROSS set.)
"""
import importlib.util
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {"raw":    "playstyle_v2_raw_v1to4.ipynb",
           "choice": "playstyle_v2_choice.ipynb",
           "both":   "playstyle_v2_both.ipynb"}

# (cell_type, source prefix that uniquely identifies the cell) of every cell to refresh
PREFIXES = [("code",     "# ===== Reviewer #3"),
            ("markdown", "### Reviewer #3"),
            ("markdown", "## 10 - Reviewer follow-ups")]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "_gen", os.path.join(HERE, "_make_three_block_notebooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find(cells, ctype, prefix, where):
    hits = [c for c in cells if c.cell_type == ctype and c.source.lstrip().startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{where}: {len(hits)} cells match ({ctype}, {prefix!r}), expected exactly 1")
    return hits[0]


def main():
    gen = _generator()
    for block, fname in TARGETS.items():
        path = os.path.join(HERE, fname)
        fresh = gen.build(block)
        nb = nbf.read(path, as_version=4)
        for ctype, prefix in PREFIXES:
            src = _find(fresh.cells, ctype, prefix, f"{fname} (generated)").source
            cell = _find(nb.cells, ctype, prefix, f"{fname} (on disk)")
            cell.source = src
            if ctype == "code":         # stale: the code that produced them is gone
                cell.outputs = []
                cell.execution_count = None
        nbf.write(nb, path)
        kept = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
        print(f"patched {fname}  ({len(PREFIXES)} cells replaced; {kept} outputs kept elsewhere)")


if __name__ == "__main__":
    main()
