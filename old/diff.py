"""Differential test: same px call on pandas / polars / pyarrow inputs."""
import base64, json, math, traceback, warnings
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import plotly
import plotly.express as px

warnings.filterwarnings("ignore")
print("plotly", plotly.__version__, "pandas", pd.__version__, "polars", pl.__version__)

DT = {"f8": "<f8", "f4": "<f4", "i1": "i1", "u1": "u1", "i2": "<i2", "u2": "<u2",
      "i4": "<i4", "u4": "<u4", "i8": "<i8", "u8": "<u8"}


def norm(o):
    if isinstance(o, dict):
        if set(o) >= {"dtype", "bdata"}:
            arr = np.frombuffer(base64.b64decode(o["bdata"]), dtype=DT[o["dtype"]])
            if "shape" in o:
                arr = arr.reshape([int(s) for s in str(o["shape"]).split(",")])
            return norm(arr.tolist())
        return {k: norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [norm(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        if isinstance(o, float) and math.isnan(o):
            return None
        return round(float(o), 6)
    return o


def fdict(fig):
    d = json.loads(fig.to_json())
    d["layout"].pop("template", None)
    return norm(d)


def diff(a, b, path=""):
    out = []
    if type(a) != type(b):
        return [f"{path}: {str(a)[:120]!r} != {str(b)[:120]!r}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'a' if k in a else 'b'} ({str(a.get(k, b.get(k)))[:100]})")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}; {str(a)[:150]} vs {str(b)[:150]}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {str(a)[:120]!r} != {str(b)[:120]!r}")
    return out


rng = np.random.default_rng(0)
n = 60
base = pd.DataFrame({
    "x": rng.normal(size=n).round(3),
    "y": rng.normal(size=n).round(3),
    "i": rng.integers(0, 10, n),
    "c": rng.choice(["a", "b", "c"], n),
    "d": rng.choice(["p", "q"], n),
    "e": rng.choice(["u", "v", "w", "z"], n),
    "t": pd.date_range("2020-01-01", periods=n, freq="D"),
    "w": rng.uniform(1, 5, n).round(2),
    "b": rng.choice([True, False], n),
})
base.loc[3, "y"] = np.nan

frames = {
    "pandas": base,
    "polars": pl.from_pandas(base),
    "pyarrow": pa.Table.from_pandas(base, preserve_index=False),
}

cases = {
    "scatter_basic": lambda df: px.scatter(df, x="x", y="y", color="c", symbol="d"),
    "scatter_facet_wrap": lambda df: px.scatter(df, x="x", y="y", facet_col="e", facet_col_wrap=3, category_orders={"e": ["z", "w", "v", "u"]}),
    "scatter_hover_dict": lambda df: px.scatter(df, x="x", y="y", hover_data={"i": ":.1f", "w": True, "c": False, "extra": ("%{y}", list(range(n)))}),
    "scatter_size": lambda df: px.scatter(df, x="x", y="y", size="w", color="i"),
    "scatter_bool_color": lambda df: px.scatter(df, x="x", y="y", color="b"),
    "scatter_trend_ols": lambda df: px.scatter(df, x="x", y="w", color="d", trendline="ols"),
    "scatter_trend_lowess": lambda df: px.scatter(df, x="x", y="w", trendline="lowess"),
    "scatter_trend_rolling_t": lambda df: px.scatter(df, x="t", y="w", trendline="rolling", trendline_options=dict(window=5)),
    "scatter_trend_ewm": lambda df: px.scatter(df, x="t", y="w", trendline="ewm", trendline_options=dict(halflife=2)),
    "scatter_trend_expanding": lambda df: px.scatter(df, x="i", y="w", trendline="expanding", trendline_options=dict(function="max")),
    "scatter_trend_ols_log": lambda df: px.scatter(df, x="w", y="w", trendline="ols", trendline_options=dict(log_x=True)),
    "scatter_marginal": lambda df: px.scatter(df, x="x", y="y", marginal_x="histogram", marginal_y="box", color="d"),
    "scatter_anim": lambda df: px.scatter(df, x="x", y="y", animation_frame="e", animation_group="c", range_x=[-3, 3]),
    "line_t": lambda df: px.line(df, x="t", y="w", color="d", markers=True, line_dash="c"),
    "area": lambda df: px.area(df, x="t", y="w", color="d"),
    "bar": lambda df: px.bar(df, x="c", y="w", color="d", text_auto=True, barmode="group"),
    "bar_pattern": lambda df: px.bar(df, x="e", y="w", pattern_shape="d"),
    "hist_avg": lambda df: px.histogram(df, x="c", y="w", histfunc="avg", color="d"),
    "hist_norm": lambda df: px.histogram(df, x="x", histnorm="probability density", nbins=10, marginal="rug"),
    "hist_cum": lambda df: px.histogram(df, x="i", cumulative=True, text_auto=".2f"),
    "ecdf": lambda df: px.ecdf(df, x="x", color="d", ecdfnorm="percent", ecdfmode="reversed"),
    "ecdf_compl_w": lambda df: px.ecdf(df, x="i", y="w", ecdfmode="complementary", ecdfnorm=None),
    "ecdf_marg": lambda df: px.ecdf(df, x="x", marginal="histogram", markers=True, lines=False),
    "box": lambda df: px.box(df, x="c", y="w", color="d", points="all", notched=True),
    "violin": lambda df: px.violin(df, y="w", x="e", box=True),
    "strip": lambda df: px.strip(df, x="c", y="w"),
    "pie": lambda df: px.pie(df, names="c", values="w"),
    "pie_nov": lambda df: px.pie(df, names="e"),
    "sunburst": lambda df: px.sunburst(df, path=["c", "d"], values="w", color="w"),
    "sunburst_cat": lambda df: px.sunburst(df, path=["c", "d", "e"], values="w", color="d"),
    "treemap": lambda df: px.treemap(df, path=[px.Constant("all"), "c", "e"], values="w"),
    "icicle": lambda df: px.icicle(df, path=["e", "c"], color="w", color_continuous_midpoint=3),
    "funnel": lambda df: px.funnel(df, x="w", y="c", color="d"),
    "funnel_area": lambda df: px.funnel_area(df, names="c", values="w"),
    "density_heatmap": lambda df: px.density_heatmap(df, x="x", y="y", z="w", histfunc="avg", facet_col="d", marginal_x="histogram"),
    "density_contour": lambda df: px.density_contour(df, x="x", y="y", color="d"),
    "scatter_matrix": lambda df: px.scatter_matrix(df, dimensions=["x", "y", "w"], color="c"),
    "parcoords": lambda df: px.parallel_coordinates(df, dimensions=["x", "y", "w"], color="i"),
    "parcats": lambda df: px.parallel_categories(df, dimensions=["c", "d"], color="w"),
    "timeline": lambda df: px.timeline(df.head(10) if hasattr(df, "head") else df.slice(0, 10), x_start="t", x_end="t", y="c"),
    "scatter_polar": lambda df: px.scatter_polar(df, r="w", theta="c", color="d"),
    "line_polar": lambda df: px.line_polar(df, r="w", theta="e", line_close=True),
    "bar_polar": lambda df: px.bar_polar(df, r="w", theta="e", color="d"),
    "scatter_ternary": lambda df: px.scatter_ternary(df, a="w", b="i", c="x"),
    "scatter_3d": lambda df: px.scatter_3d(df, x="x", y="y", z="w", color="c"),
    "line_3d": lambda df: px.line_3d(df, x="x", y="y", z="w"),
    "scatter_geo": lambda df: px.scatter_geo(df, lat="x", lon="y", color="c"),
    "wide_line": lambda df: px.line(df, x="t", y=["w", "x"]),
    "wide_bar_none_x": lambda df: px.bar(df, y=["w", "i"]),
    "hist_wide": lambda df: px.histogram(df, x=["w", "x"], barmode="overlay"),
    "box_wide": lambda df: px.box(df, y=["w", "x", "y"]),
    "facet_row_col": lambda df: px.scatter(df, x="x", y="y", facet_row="d", facet_col="c", facet_row_spacing=0.1),
    "line_group": lambda df: px.line(df, x="i", y="w", line_group="c", color="d", text="e"),
    "symbol_map": lambda df: px.scatter(df, x="x", y="y", symbol="c", symbol_map={"a": "square", "b": "diamond"}),
    "color_map": lambda df: px.bar(df, x="c", y="w", color="c", color_discrete_map={"a": "red", "b": "blue", "c": "green"}),
    "color_map_identity": lambda df: px.scatter(df.head(3) if hasattr(df, "head") else df.slice(0, 3), x="x", y="y", color="c"),
    "custom_data": lambda df: px.scatter(df, x="x", y="y", custom_data=["c", "i"], hover_name="e"),
    "error_bars": lambda df: px.scatter(df, x="x", y="y", error_y="w", error_x_minus="w"),
    "bar_base": lambda df: px.bar(df, x="c", y="w", base="i"),
    "log_axes": lambda df: px.scatter(df, x="w", y="w", log_x=True, range_x=[1, 5]),
    "labels": lambda df: px.scatter(df, x="x", y="y", color="c", labels={"x": "XX", "c": "CC"}, facet_col="d"),
}

for name, fn in cases.items():
    res = {}
    for k, df in frames.items():
        try:
            res[k] = fdict(fn(df))
        except Exception as ex:
            res[k] = "ERR " + type(ex).__name__ + ": " + str(ex).splitlines()[0][:200]
    for k in ("polars", "pyarrow"):
        a, b = res["pandas"], res[k]
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                print(f"[{name}] pandas vs {k}:", (a if isinstance(a, str) else "OK"), "|", (b if isinstance(b, str) else "OK"))
            continue
        d = diff(a, b)
        if d:
            print(f"[{name}] pandas vs {k}: {len(d)} diffs")
            for line in d[:6]:
                print("    ", line)
    if all(not isinstance(v, str) for v in res.values()) and name:
        pass
print("DONE diff")
