import json, traceback, warnings, datetime, decimal, os, tempfile
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.io.json import to_json_plotly
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
print("plotly", plotly.__version__)


def probe(fn):
    print(f"--- {fn.__name__}")
    try:
        fn()
    except Exception as e:
        print("EXC", type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "")
        traceback.print_exc(limit=-2)


def first_diff(a, b, p=""):
    if type(a) != type(b):
        return f"{p}: {str(a)[:200]} || {str(b)[:200]}"
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                return f"{p}.{k} only in one: {str(a.get(k, b.get(k)))[:200]}"
            r = first_diff(a[k], b[k], p + "." + k)
            if r:
                return r
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{p} len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = first_diff(x, y, f"{p}[{i}]")
            if r:
                return r
    elif a != b:
        return f"{p}: {a!r} || {b!r}"
    return None


@probe
def json_roundtrip_detail():
    for name, fig in [
        ("go_simple", go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))),
        ("go_np", go.Figure(go.Scatter(x=np.arange(3), y=np.arange(3) * 1.5))),
        ("px", px.scatter(x=np.arange(5), y=np.arange(5) * 1.5, color=list("aabbc"))),
        ("hline", go.Figure().add_hline(y=2, annotation_text="h")),
        ("heat2d", go.Figure(go.Heatmap(z=np.arange(6).reshape(2, 3)))),
    ]:
        s = fig.to_json()
        fig2 = pio.from_json(s)
        s2 = fig2.to_json()
        print(name, "equal:", s == s2, first_diff(json.loads(s), json.loads(s2)))


@probe
def json_engines():
    objs = {
        "dt64arr_nat": np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"),
        "series_tz_nat": pd.Series([pd.Timestamp("2020-01-01 05:00", tz="US/Eastern"), pd.NaT]),
        "dt64_nat": np.datetime64("NaT"),
        "pdNA": pd.NA,
        "np_scalars": [np.float32(1.5), np.int8(3), np.bool_(True)],
        "date": datetime.date(2020, 1, 2),
        "time": datetime.time(1, 2, 3),
        "tz_dt": datetime.datetime(2020, 1, 1, 1, tzinfo=datetime.timezone.utc),
        "pdts_tz": pd.Timestamp("2020-01-01 05:00", tz="US/Eastern"),
        "decimal": decimal.Decimal("1.25"),
        "nan_inf": [float("nan"), float("inf"), np.nan],
        "np_nan_arr": np.array([1.0, np.nan, np.inf]),
        "dict_intkey": {1: "a", np.int64(2): "b"},
        "obj_arr": np.array([1, "a", None], dtype=object),
        "pd_index": pd.Index([1, 2]),
        "pd_dtindex": pd.DatetimeIndex(["2020-01-01", None]),
        "timedelta": pd.Timedelta("1D"),
        "str_ext": pd.array(["a", None], dtype="string"),
        "u2028": "a b</script>",
        "big_int": 2**70,
        "pl_series": pl.Series([1, 2]),
        "np_str": np.str_("x"),
        "pd_period": pd.Period("2020-01", freq="M"),
        "cat_series": pd.Series(["a", "b"], dtype="category"),
        "float_series_nan": pd.Series([1.0, None]),
        "int_series_na": pd.Series([1, None], dtype="Int64"),
    }
    for k, v in objs.items():
        out = []
        for eng in ["json", "orjson"]:
            try:
                out.append(to_json_plotly({"v": v}, engine=eng))
            except Exception as e:
                out.append("ERR " + type(e).__name__ + ": " + str(e)[:100])
        flag = "" if out[0] == out[1] else "   <<< DIFF"
        print(k, "|", out[0][:150], "|", out[1][:150], flag)


@probe
def trendline_same_xy():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    for lib, d in [("pandas", df), ("polars", pl.from_pandas(df)), ("pyarrow", pa.Table.from_pandas(df))]:
        try:
            f = px.scatter(d, x="a", y="a", trendline="ols")
            print(lib, "ok", list(f.data[1].y))
        except Exception as e:
            print(lib, "ERR", type(e).__name__, str(e)[:150])
    for lib, d in [("pandas", df), ("polars", pl.from_pandas(df))]:
        for kw in [dict(x="a", y="a", color="a"), dict(x="a", y="a", hover_data=["a"]), dict(x="a", y="a", marginal_x="histogram")]:
            try:
                f = px.scatter(d, **kw)
                print(lib, kw, "ok")
            except Exception as e:
                print(lib, kw, "ERR", type(e).__name__, str(e)[:150])


@probe
def nulls_in_groupers():
    pdf = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "g": ["a", None, "b", "a"]})
    for lib, d in [("pandas", pdf), ("polars", pl.from_pandas(pdf)), ("pyarrow", pa.Table.from_pandas(pdf))]:
        for kw in [dict(color="g"), dict(facet_col="g"), dict(symbol="g"), dict(animation_frame="g")]:
            try:
                f = px.scatter(d, x="x", y="y", **kw)
                npts = sum(len(t.x) for t in f.data)
                print(lib, kw, "traces", [t.name for t in f.data], "pts", npts, "ann", [a.text for a in f.layout.annotations], "frames", [fr.name for fr in f.frames])
            except Exception as e:
                print(lib, kw, "ERR", type(e).__name__, str(e)[:150])


@probe
def categorical_dtypes():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": pd.Categorical(["b", "a", "b"], categories=["c", "b", "a"], ordered=True)})
    f = px.scatter(pdf, x="x", y="y", color="g")
    print("pandas cat traces", [t.name for t in f.data])
    f = px.bar(pdf, x="g", y="y")
    print("pandas cat bar x", list(f.data[0].x), f.layout.xaxis.categoryarray)
    pldf = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": pl.Series(["b", "a", "b"], dtype=pl.Enum(["c", "b", "a"]))})
    f = px.scatter(pldf, x="x", y="y", color="g")
    print("polars enum traces", [t.name for t in f.data])


@probe
def bool_color_map():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "b": [True, False, True]})
    for lib, d in [("pandas", df), ("polars", pl.from_pandas(df))]:
        f = px.scatter(d, x="x", y="y", color="b", color_discrete_map={True: "red", False: "blue"}, category_orders={"b": [False, True]})
        print(lib, [(t.name, t.marker.color) for t in f.data])


@probe
def numeric_category_orders():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "g": [1, 2, 1, 3]})
    for lib, d in [("pandas", df), ("polars", pl.from_pandas(df))]:
        f = px.scatter(d, x="x", y="y", facet_col="g", category_orders={"g": [3, 1, 2]})
        print(lib, [a.text for a in f.layout.annotations], [(t.xaxis, list(t.x)) for t in f.data])
        f = px.bar(d, x="g", y="y", color="g", category_orders={"g": [3, 1, 2]}, color_discrete_sequence=["r", "g", "b"][:0] or None)
        print(lib, [t.name for t in f.data])


@probe
def hover_data_forms():
    df = px.data.tips()
    f = px.scatter(df, x="total_bill", y="tip", color="sex", hover_data={"sex": False, "size": ":.2f", "day": True, "tip": ":$.2f", "new": [str(i) for i in range(len(df))]})
    print(f.data[0].hovertemplate)
    f = px.scatter(df, x="total_bill", y="tip", hover_data={"total_bill": False, "rnd": (":.1f", np.random.default_rng(0).random(len(df)))})
    print(f.data[0].hovertemplate)
    f = px.bar(df, x="day", y="tip", hover_data={"size": True}, custom_data=["size", "sex"])
    print(f.data[0].hovertemplate, np.asarray(f.data[0].customdata)[:1])
    f = px.histogram(df, x="total_bill", hover_data=["tip"])
    print("hist", f.data[0].hovertemplate)
    f = px.scatter(df, x="total_bill", y="tip", hover_name="day", hover_data={"day": True})
    print(f.data[0].hovertemplate)


@probe
def timeline_variants():
    pdf = pd.DataFrame({"s": pd.to_datetime(["2020-01-01", "2020-02-01"]), "e": pd.to_datetime(["2020-01-15", "2020-03-01"]), "t": ["A", "B"]})
    for lib, d in [("pandas", pdf), ("polars", pl.from_pandas(pdf)), ("polars_date", pl.from_pandas(pdf).with_columns(pl.col("s", "e").cast(pl.Date)))]:
        try:
            f = px.timeline(d, x_start="s", x_end="e", y="t", color="t")
            print(lib, [(list(t.base), list(t.x)) for t in f.data])
        except Exception as e:
            print(lib, "ERR", type(e).__name__, str(e)[:200])


@probe
def size_with_nan():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "s": [1.0, np.nan, 3.0]})
    f = px.scatter(df, x="x", y="y", size="s")
    print(f.data[0].marker.size, f.data[0].marker.sizeref)


@probe
def annotations_domain_subplots():
    f = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]])
    f.add_pie(values=[1, 2], row=1, col=1)
    f.add_scatter(y=[1, 2], row=1, col=2)
    try:
        f.add_annotation(text="x", row=1, col=2)
        print([a.to_plotly_json() for a in f.layout.annotations])
    except Exception as e:
        print("ERR", e)
    f.update_annotations(font_size=20)
    print(len(list(f.select_annotations(row=1, col=2))))


@probe
def select_annotations_px_facets():
    df = px.data.tips()
    f = px.scatter(df, x="total_bill", y="tip", facet_col="day", facet_row="sex")
    f.add_annotation(text="hi", x=20, y=5, row=2, col=3)
    print([(a.text, a.xref, a.yref) for a in f.select_annotations(row=2, col=3)])


@probe
def write_html_directory():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sub", "f.html")
    os.makedirs(os.path.dirname(p))
    go.Figure().write_html(p, include_plotlyjs="directory")
    print(sorted(os.listdir(os.path.dirname(p))))
    go.Figure().write_html(p, include_plotlyjs="cdn", auto_open=False, full_html=False, validate=False)
    print(open(p).read()[:200])


@probe
def subplots_shared_yaxes_columns_secondary():
    f = make_subplots(rows=1, cols=2, shared_yaxes=True, specs=[[{"secondary_y": True}, {"secondary_y": True}]])
    f.add_scatter(y=[1, 2], row=1, col=1, secondary_y=True)
    f.add_scatter(y=[1, 2], row=1, col=2, secondary_y=True)
    print({k: v for k, v in f.layout.to_plotly_json().items() if "axis" in k})


@probe
def subplots_shared_all():
    f = make_subplots(rows=2, cols=2, shared_xaxes="all", shared_yaxes="rows")
    print({k: v for k, v in f.layout.to_plotly_json().items() if "axis" in k})
    f = make_subplots(rows=2, cols=2, shared_xaxes="columns", start_cell="bottom-left")
    print({k: v for k, v in f.layout.to_plotly_json().items() if "axis" in k})


@probe
def insets():
    f = make_subplots(insets=[{"cell": (1, 1), "l": 0.7, "b": 0.3}], subplot_titles=["main", "inset"])
    print({k: v for k, v in f.layout.to_plotly_json().items() if "axis" in k}, [a.text for a in f.layout.annotations])


@probe
def range_and_log_facets():
    df = px.data.tips()
    f = px.scatter(df, x="total_bill", y="tip", facet_col="day", facet_col_wrap=2, log_x=True, range_x=[1, 100])
    print([(a.plotly_name, a.type, a.range) for a in f.select_xaxes()])


@probe
def imshow_binary_facets():
    img = (np.random.default_rng(0).random((3, 4, 5, 3)) * 255).astype(np.uint8)
    f = px.imshow(img, facet_col=0, binary_string=True)
    print(len(f.data), [t.type for t in f.data], [a.text for a in f.layout.annotations])
    f = px.imshow(img[0, :, :, 0], color_continuous_scale="gray", binary_string=True, origin="lower")
    print(f.data[0].type, f.layout.yaxis.autorange)
    f = px.imshow(np.array([[1, np.nan], [3, 4]]), text_auto=True)
    print(f.data[0].texttemplate)


@probe
def histogram_labels():
    df = px.data.tips()
    f = px.histogram(df, x="day", y="tip", histfunc="avg", labels={"tip": "Tip"})
    print(f.layout.yaxis.title.text, f.data[0].hovertemplate)
    f = px.histogram(df, x="day", y="tip", histfunc="sum", histnorm="percent", orientation="v")
    print(f.layout.yaxis.title.text)
    f = px.histogram(df, y="day", x="tip", histfunc="avg", barnorm="fraction", color="sex")
    print(f.layout.xaxis.title.text)


@probe
def ecdf_tie_values():
    f = px.ecdf(x=[1, 1, 2], markers=True)
    print(list(f.data[0].x), list(f.data[0].y))


@probe
def area_line_group_sort():
    df = pd.DataFrame({"x": [3, 1, 2], "y": [1, 2, 3]})
    f = px.line(df, x="x", y="y")
    print(list(f.data[0].x))


@probe
def scatter_matrix_labels():
    df = px.data.iris()
    f = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length"], labels={"sepal_width": "SW"}, color="species", symbol="species")
    print([d.label for d in f.data[0].dimensions], f.data[0].hovertemplate)


@probe
def color_continuous_midpoint_and_range():
    df = px.data.tips()
    f = px.scatter(df, x="total_bill", y="tip", color="size", range_color=[0, 10], color_continuous_midpoint=3)
    print(f.layout.coloraxis.cmin, f.layout.coloraxis.cmax, f.layout.coloraxis.cmid)


@probe
def update_layout_template_string_merge():
    f = go.Figure(layout_template="plotly_dark+presentation")
    print(f.layout.template.layout.paper_bgcolor, f.layout.template.layout.font.size)
    pio.templates.default = "plotly_white+none"
    f = go.Figure()
    print(f.layout.template.layout.paper_bgcolor)
    pio.templates.default = "plotly"


@probe
def figure_dict_roundtrip():
    f = px.scatter(px.data.iris(), x="sepal_width", y="sepal_length", animation_frame="species")
    f2 = go.Figure(f.to_dict())
    print("frames", len(f2.frames), f2.to_json() == f.to_json())
    f3 = go.Figure(f)
    print(f3.to_json() == f.to_json())
    f4 = pio.from_json(f.to_json(), output_type="FigureWidget", skip_invalid=True) if False else None


@probe
def batch_update_selectors():
    f = make_subplots(rows=2, cols=2)
    for r in (1, 2):
        for c in (1, 2):
            f.add_scatter(y=[r, c], row=r, col=c, name=f"{r}{c}")
    f.update_traces(marker_color="red", row=[1, 2] if False else 2)
    print([(t.name, t.marker.color) for t in f.data])
    f.for_each_xaxis(lambda a: a.update(title_text="T"), row=1)
    print([(a.plotly_name, a.title.text) for a in f.select_xaxes()])
    print([t.name for t in f.select_traces(selector=lambda t: t.name.startswith("1"))])
    print([t.name for t in f.select_traces(selector=2)])
    print([t.name for t in f.select_traces(selector=-1)])
