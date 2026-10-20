#!/usr/bin/env python3
"""Execute code cells in Jupyter Notebooks and populate output fields."""

import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def execute_notebook(nb_path: Path):
    print(f"Executing {nb_path}...")
    with open(nb_path) as f:
        nb = json.load(f)

    namespace = {}
    exec_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code_text = "".join(cell["source"])
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        outputs = []

        # Clear matplotlib figures before running
        plt.close("all")

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code_text, namespace)

            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()

            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out_text.splitlines()]
                })
            if err_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_text.splitlines()]
                })

            # Check if any matplotlib figures were generated
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                b64_data = base64.b64encode(img_buf.read()).decode("ascii")
                outputs.append({
                    "data": {
                        "image/png": b64_data,
                        "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100:.0f}x{fig.get_size_inches()[1]*100:.0f} with {len(fig.axes)} Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in cell {idx} of {nb_path}: {e}")
            outputs.append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb.splitlines()
            })

        cell["execution_count"] = exec_count
        cell["outputs"] = outputs
        exec_count += 1

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully executed and saved {nb_path}")


if __name__ == "__main__":
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    for nb in notebooks:
        execute_notebook(nb)
