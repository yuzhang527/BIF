"""SwanLab experiment tracking integration.

Two integration paths:
  1. Manual: init_run() + log() + finish() — used by all modules
  2. Pipeline: shared SwanLab run via env vars (SWANLAB_PIPELINE_RUN_ID)
"""

from __future__ import annotations

import os
import time
from typing import Any

import swanlab

_pending_logs: list[tuple[dict[str, Any], int | None]] = []
_INIT_TIMEOUT = 30

_ENV_RUN_ID = "SWANLAB_PIPELINE_RUN_ID"
_ENV_EXPERIMENT = "SWANLAB_PIPELINE_EXPERIMENT"
_ENV_PROJECT = "SWANLAB_PROJECT"

_DEFAULT_PROJECT = "BIFrost"

_PALETTE = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#8e44ad", "#27ae60",
    "#d35400", "#c0392b", "#16a085", "#2c3e50", "#f1c40f",
    "#7f8c8d", "#d63384", "#0d6efd", "#198754", "#ffc107",
]

_chart_color_counter: int = 0
_chart_color_map: dict[str, int] = {}


def _key_base(key: str) -> int:
    """Deterministic color index — each unique chart path gets the next color."""
    global _chart_color_counter
    if key not in _chart_color_map:
        _chart_color_map[key] = _chart_color_counter
        _chart_color_counter += 1
    return _chart_color_map[key] % len(_PALETTE)


def get_project() -> str:
    return os.environ.get(_ENV_PROJECT, _DEFAULT_PROJECT)


def _is_initialised() -> bool:
    try:
        return swanlab.get_run() is not None
    except Exception:
        return False


def _flush_pending() -> None:
    global _pending_logs
    if not _pending_logs:
        return
    remaining: list[tuple[dict[str, Any], int | None]] = []
    for data, step in _pending_logs:
        try:
            swanlab.log(data, step=step)
        except Exception:
            remaining.append((data, step))
    _pending_logs = remaining


def init_run(
    experiment_name: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    description: str = "",
    run_name: str | None = None,
    project: str | None = None,
) -> None:
    """Initialize a SwanLab run (call only from rank 0)."""
    global _pending_logs
    _pending_logs = []

    pipeline_run_id = os.environ.get(_ENV_RUN_ID)
    pipeline_experiment = os.environ.get(_ENV_EXPERIMENT)

    _proj = project if project is not None else get_project()
    if pipeline_run_id and pipeline_experiment:
        display_name = run_name if run_name is not None else experiment_name
        init_kwargs: dict[str, Any] = {
            "project": _proj,
            "experiment_name": display_name,
            "group": pipeline_experiment,
            "description": description,
            "config": config,
            "tags": tags,
        }
    else:
        init_kwargs = {
            "project": _proj,
            "experiment_name": experiment_name,
            "group": run_name if run_name is not None else None,
            "description": f"run_name={run_name}" if run_name else description,
            "config": config,
            "tags": tags,
        }

    swanlab.init(**init_kwargs)
    deadline = time.monotonic() + _INIT_TIMEOUT
    while time.monotonic() < deadline:
        if _is_initialised():
            break
        time.sleep(0.5)



def log(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics. Buffers calls made before init completes."""
    if not _is_initialised():
        _pending_logs.append((data, step))
        return
    _flush_pending()
    try:
        swanlab.log(data, step=step)
    except Exception:
        pass


def finish() -> None:
    """Close the SwanLab run."""
    if not _is_initialised():
        return
    _flush_pending()
    try:
        swanlab.finish()
    except Exception:
        pass


def finish_pipeline() -> None:
    """Force-close the SwanLab run at the end of a pipeline.

    Unlike finish(), this ALWAYS closes the run even in pipeline mode.
    """
    if not _is_initialised():
        return
    _flush_pending()
    try:
        swanlab.finish()
    except Exception:
        pass


def log_image(key: str, path: str) -> None:
    """Log an image from a file path.  Prefer log_figure() for matplotlib figures."""
    if not _is_initialised():
        return
    try:
        swanlab.log({key: swanlab.Image(path)})
    except Exception:
        pass


def log_figure(key: str, fig: "matplotlib.figure.Figure") -> None:  # noqa: F821
    """Log a matplotlib Figure directly to SwanLab without saving to disk.

    SwanLab accepts ``swanlab.Image`` constructed from a PIL image, so we
    render the figure into an in-memory PNG buffer and hand it off — no
    temporary file is created.

    Args:
        key:  The metric name shown in SwanLab (e.g. ``"loss_curve"``).
        fig:  A matplotlib Figure object.  The figure is closed after logging
              so callers do not need to call ``plt.close()`` themselves.
    """
    if not _is_initialised():
        return
    try:
        import io
        import matplotlib  # noqa: F401 — ensure it is importable
        from PIL import Image as PILImage

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()  # copy to detach from buffer
        buf.close()
        swanlab.log({key: swanlab.Image(pil_img)})
    except Exception:
        pass
    finally:
        try:
            import matplotlib.pyplot as _plt

            _plt.close(fig)
        except Exception:
            pass


def log_heatmap(
    key: str,
    xaxis: list[str],
    yaxis: list[str],
    matrix: "np.ndarray",  # noqa: F821  shape (len(yaxis), len(xaxis))
    value_label: str = "value",
    precision: int = 4,
    show_labels: bool = False,
) -> None:
    """Log a 2-D matrix as a native SwanLab echarts HeatMap.

    Args:
        key:         Metric name shown in SwanLab.
        xaxis:       Labels for the x-axis (columns).
        yaxis:       Labels for the y-axis (rows).
        matrix:      2-D array of shape (len(yaxis), len(xaxis)).
        value_label: Series name displayed in the tooltip.
        precision:   Decimal places to round values to.
        show_labels: If True, show numeric labels on cells (default False for readability).
    """
    if not _is_initialised():
        return
    try:
        from pyecharts.options.series_options import LabelOpts

        value = [
            [j, i, round(float(matrix[i, j]), precision)]
            for i in range(len(yaxis))
            for j in range(len(xaxis))
        ]
        chart = swanlab.echarts.HeatMap()
        chart.add_xaxis(xaxis)
        chart.add_yaxis(
            value_label,
            yaxis,
            value,
            label_opts=LabelOpts(is_show=show_labels),
        )
        chart.set_global_opts(
            visualmap_opts={"min": round(float(matrix.min()), precision),
                            "max": round(float(matrix.max()), precision),
                            "calculable": True},
        )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_bar(
    key: str,
    xaxis: list[str],
    series: dict[str, list],
    stack: bool = False,
) -> None:
    if not _is_initialised():
        return
    try:
        from pyecharts.options.series_options import ItemStyleOpts, LabelOpts

        base = _key_base(key)
        chart = swanlab.echarts.Bar()
        chart.add_xaxis(xaxis)
        for idx, (name, values) in enumerate(series.items()):
            color = _PALETTE[(base + idx) % len(_PALETTE)]
            chart.add_yaxis(
                name,
                [round(float(v), 4) if v is not None else None for v in values],
                stack="stack0" if stack else None,
                label_opts=LabelOpts(is_show=False),
                itemstyle_opts=ItemStyleOpts(color=color),
            )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_line(
    key: str,
    xaxis: list[str],
    series: dict[str, list],
    smooth: bool = False,
) -> None:
    if not _is_initialised():
        return
    try:
        from pyecharts.options.global_options import (
            ToolboxOpts,
            ToolBoxFeatureOpts,
            ToolBoxFeatureDataZoomOpts,
            ToolBoxFeatureMagicTypeOpts,
            ToolBoxFeatureRestoreOpts,
        )
        from pyecharts.options.series_options import ItemStyleOpts

        chart = swanlab.echarts.Line()
        chart.add_xaxis(xaxis)
        base = _key_base(key)
        for idx, (name, values) in enumerate(series.items()):
            color = _PALETTE[(base + idx) % len(_PALETTE)]
            chart.add_yaxis(
                name,
                [round(float(v), 6) if v is not None else None for v in values],
                is_smooth=smooth,
                is_symbol_show=False,
                itemstyle_opts=ItemStyleOpts(color=color),
            )
        chart.set_global_opts(
            toolbox_opts=ToolboxOpts(
                feature=ToolBoxFeatureOpts(
                    magic_type=ToolBoxFeatureMagicTypeOpts(
                        type_=["line", "bar", "stack"],
                    ),
                    data_zoom=ToolBoxFeatureDataZoomOpts(
                        is_show=True,
                    ),
                    restore=ToolBoxFeatureRestoreOpts(
                        is_show=True,
                    ),
                ),
            ),
        )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_scatter(
    key: str,
    xaxis_name: str,
    yaxis_name: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    if not _is_initialised():
        return
    try:
        from pyecharts.options.series_options import ItemStyleOpts, LabelOpts

        chart = swanlab.echarts.Scatter()
        chart.add_xaxis([])
        base = _key_base(key)
        for idx, (name, points) in enumerate(series.items()):
            color = _PALETTE[(base + idx) % len(_PALETTE)]
            data = [[round(float(x), 6), round(float(y), 6)] for x, y in points]
            chart.add_yaxis(
                name,
                data,
                symbol_size=5,
                label_opts=LabelOpts(is_show=False),
                itemstyle_opts=ItemStyleOpts(color=color),
            )
        chart.set_global_opts(
            xaxis_opts={"type": "value", "name": xaxis_name},
            yaxis_opts={"type": "value", "name": yaxis_name},
        )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_pie(
    key: str,
    series_name: str,
    data: list[tuple[str, float]],
) -> None:
    """Log a pie chart as a native SwanLab echarts Pie.

    Args:
        key:         Metric name shown in SwanLab.
        series_name: Series name displayed in the tooltip.
        data:        List of (label, value) tuples.
    """
    if not _is_initialised():
        return
    try:
        chart = swanlab.echarts.Pie()
        chart.add(
            series_name,
            data,
            radius=["30%", "70%"],
        )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_boxplot(
    key: str,
    xaxis: list[str],
    series: dict[str, list[list[float]]],
) -> None:
    """Log a boxplot as a native SwanLab echarts Boxplot.

    Each series maps a name to a list of 5-number summaries
    [min, Q1, median, Q3, max] per category.

    Args:
        key:    Metric name shown in SwanLab.
        xaxis:  Category labels for the x-axis.
        series: Dict mapping series name → list of [min, Q1, med, Q3, max].
    """
    if not _is_initialised():
        return
    try:
        from pyecharts.options.series_options import ItemStyleOpts

        chart = swanlab.echarts.Boxplot()
        chart.add_xaxis(xaxis)
        base = _key_base(key)
        for idx, (name, boxes) in enumerate(series.items()):
            color = _PALETTE[(base + idx) % len(_PALETTE)]
            chart.add_yaxis(name, boxes, itemstyle_opts=ItemStyleOpts(color=color))
        swanlab.log({key: chart})
    except Exception:
        pass


def log_table(
    key: str,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    """Log a table as a native SwanLab echarts Table.

    Useful for displaying top-K samples with text previews, scores,
    and metadata.

    Args:
        key:     Metric name shown in SwanLab (e.g. ``"top_samples"``).
        headers: Column names.
        rows:    List of rows, each row is a list of values matching headers.
    """
    if not _is_initialised():
        return
    try:
        table = swanlab.echarts.Table()
        str_rows = []
        for row in rows:
            str_rows.append([str(v) if v is not None else "" for v in row])
        table.add(headers=headers, rows=str_rows)
        swanlab.log({key: table})
    except Exception:
        pass
