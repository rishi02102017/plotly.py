import json, traceback, warnings
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
print("plotly", plotly.__version__)


def probe(fn):
    print(f"--- {fn.__name__}")
    try:
        fn()
    except Exception:
        traceback.print_exc(limit=3)


@probe
def table_font_colors_df():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    cols5 = ["red", "green", "blue", "orange", "purple"]  # header + 4 rows
    try:
        f = ff.create_table(df, font_colors=cols5)
        print("5 colors ok", [a.font.color for a in f.layout.annotations][::2])
    except Exception as e:
        print("5 colors ERR", e)
    cols4 = cols5[:4]
    f = ff.create_table(df, font_colors=cols4)
    print("4 colors ->", [a.font.color for a in f.layout.annotations][::2])
    lst = [["a", "b"]] + df.values.tolist()
    f = ff.create_table(lst, font_colors=cols5)
    print("list 5 colors ->", [a.font.color for a in f.layout.annotations][::2])


@probe
def table_index_shared_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
    f = ff.create_table(df, index=True, index_title="idx")
    print("z", f.data[0].z)
    print("texts", [a.text for a in f.layout.annotations])


@probe
def json_roundtrip():
    fig = px.scatter(x=np.arange(5), y=np.arange(5) * 1.5, color=list("aabbc"))
    fig.add_hline(y=2, annotation_text="h")
    s = fig.to_json()
    fig2 = pio.from_json(s)
    print("equal json", fig2.to_json() == s)
    for eng in ["json", "orjson"]:
        try:
            print(eng, pio.to_json({"a": np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"),
                                    "b": pd.Series([pd.Timestamp("2020-01-01", tz="UTC"), pd.NaT]),
                                    "c": np.datetime64("NaT"), "d": pd.NA, "e": [np.float32(1.5), np.int8(3)]}, engine=eng))
        except Exception as e:
            print(eng, "ERR", type(e).__name__, e)


@probe
def subplots_row_heights_bottom_left():
    f = make_subplots(rows=3, cols=1, row_heights=[0.6, 0.3, 0.1], start_cell="bottom-left", vertical_spacing=0)
    print([f.layout[f"yaxis{i if i > 1 else ''}"].domain for i in range(1, 4)])
    f = make_subplots(rows=1, cols=3, column_widths=[1, 2, 3])
    print([f.layout[f"xaxis{i if i > 1 else ''}"].domain for i in range(1, 4)])


@probe
def subplots_titles_rowspan():
    f = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]], subplot_titles=["A", "B", "C"])
    print([(a.text, round(a.x, 3), round(a.y, 3)) for a in f.layout.annotations])
    f = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], subplot_titles=["A", "B", "C"], start_cell="bottom-left")
    print([(a.text, round(a.x, 3), round(a.y, 3)) for a in f.layout.annotations])


@probe
def subplots_shared_secondary():
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    print({k: (v.matches, v.anchor, v.overlaying) for k, v in f.layout.to_plotly_json().items() if False})
    print(json.dumps({k: v for k, v in f.layout.to_plotly_json().items() if "axis" in k}))


@probe
def select_secondary():
    f = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}]])
    f.add_scatter(y=[1, 2], row=1, col=1)
    f.add_scatter(y=[1, 2], row=1, col=1, secondary_y=True)
    f.add_scatter(y=[1, 2], row=1, col=2, secondary_y=True)
    print("sec traces", [t.yaxis for t in f.select_traces(secondary_y=True)])
    print("prim traces", [t.yaxis for t in f.select_traces(secondary_y=False)])
    print("yaxes sec", [a.plotly_name for a in f.select_yaxes(secondary_y=True)])
    print("yaxes col2", [a.plotly_name for a in f.select_yaxes(col=2)])


@probe
def vline_dates_subplots():
    df = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=5), "y": range(5), "f": list("aabbb")})
    fig = px.line(df, x="t", y="y", facet_col="f")
    fig.add_vline(x=pd.Timestamp("2021-01-03"), annotation_text="v", annotation_position="top left")
    fig.add_vrect(x0="2021-01-02", x1="2021-01-04", annotation_text="r", annotation_position="bottom right")
    print([s.to_plotly_json() for s in fig.layout.shapes])
    print([a.to_plotly_json() for a in fig.layout.annotations if a.text in ("v", "r")])


@probe
def hline_log():
    fig = px.scatter(x=[1, 10, 100], y=[1, 10, 100], log_y=True)
    fig.add_hrect(y0=5, y1=50, annotation_text="r", annotation_position="inside top")
    fig.add_hline(y=10, annotation_text="h", annotation_position="bottom")
    print([a.to_plotly_json() for a in fig.layout.annotations])


@probe
def for_each_and_update_overwrite():
    fig = go.Figure(go.Scatter(y=[1], marker=dict(color="red", size=5)))
    fig.update_traces(marker=dict(size=10), overwrite=True)
    print(fig.data[0].marker)
    fig.update_layout(dict(xaxis_title_text="a"), xaxis_range=[0, 1])
    print(fig.layout.xaxis)


@probe
def facet_wrap_row_select():
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", facet_col="day", facet_col_wrap=3)
    fig.add_hline(y=5, row=1, col=1)
    print([(s.xref, s.yref) for s in fig.layout.shapes])
    fig.update_xaxes(title_text="Q", row=1)
    print({k: v.title.text for k, v in fig.layout.to_plotly_json().items() if False})
    print([(a.plotly_name, a.title.text) for a in fig.select_xaxes()])


@probe
def write_html_opts():
    fig = go.Figure(go.Scatter(y=[1, 2]))
    h = pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id="mydiv", default_width="50%", default_height=300, config={"responsive": False}, auto_play=False)
    print(h[:600])
    h = pio.to_html(fig, include_mathjax="cdn", post_script="console.log('{plot_id}')", div_id="zz")
    print("postscript ok", "console.log('zz')" in h)


@probe
def templates_merge():
    pio.templates["t1"] = go.layout.Template(layout=dict(font_color="red"))
    pio.templates["t2"] = go.layout.Template(layout=dict(paper_bgcolor="blue"))
    fig = go.Figure(layout_template="t1+t2")
    print(fig.layout.template.layout.font, fig.layout.template.layout.paper_bgcolor)
    print(pio.templates.merge_templates("t1", "t2").layout)


@probe
def imshow_facets():
    img = np.random.default_rng(0).random((4, 5, 6))
    fig = px.imshow(img, facet_col=0, facet_col_wrap=2, labels={"facet_col": "k"}, facet_col_spacing=0.05)
    print([a.text for a in fig.layout.annotations])
    fig = px.imshow(img, animation_frame=0, text_auto=".1f", zmin=0, zmax=1)
    print(len(fig.frames), fig.data[0].texttemplate, fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax)
    fig = px.imshow(np.array([[1, 2], [3, 4]]), x=["a", "b"], y=["c", "d"], origin="lower", aspect="auto")
    print(fig.data[0].x, fig.data[0].y, fig.layout.yaxis.autorange)


@probe
def imshow_xarray_like_df():
    df = pd.DataFrame([[1, 2], [3, 4]], index=["r1", "r2"], columns=["c1", "c2"])
    fig = px.imshow(df, labels=dict(x="X", y="Y", color="Z"))
    print(fig.data[0].x, fig.data[0].y, fig.data[0].hovertemplate)


@probe
def quiver_streamline():
    x, y = np.meshgrid(np.arange(0, 2, 0.5), np.arange(0, 2, 0.5))
    f = ff.create_quiver(x, y, np.cos(x), np.sin(y), scale=0.2, name="q")
    print(len(f.data), f.data[0].name)
    xs = np.linspace(-1, 1, 10)
    f = ff.create_streamline(xs, xs, np.ones((10, 10)), np.ones((10, 10)))
    print(len(f.data))


@probe
def dendrogram_labels_hover():
    X = np.random.default_rng(1).random((6, 3))
    f = ff.create_dendrogram(X, labels=list("abcdef"), orientation="left", color_threshold=0.5)
    print(f.layout.yaxis.ticktext, f.layout.yaxis.tickvals)
    print(sorted(set(t.marker.color for t in f.data)))


@probe
def px_range_facets():
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex", range_x=[0, 60], range_y=[0, 12], log_x=False)
    print([(a.plotly_name, a.range) for a in fig.select_xaxes()], [(a.plotly_name, a.range) for a in fig.select_yaxes()])


@probe
def px_category_orders_partial():
    df = px.data.tips()
    fig = px.bar(df, x="day", y="tip", color="time", category_orders={"day": ["Sun"]})
    print(fig.layout.xaxis.categoryarray, [t.name for t in fig.data])
    fig = px.histogram(df, x="day", category_orders={"day": ["Sun", "Sat", "Fri", "Thur"]}, facet_col="smoker", facet_col_wrap=1)
    print(fig.layout.xaxis.categoryarray)


@probe
def px_color_seq_by_map():
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", color="day", color_discrete_map="identity")
    print([t.marker.color for t in fig.data][:3])


@probe
def px_hover_data_dupe():
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", hover_data=["tip", "size"], hover_name="day", custom_data=["size"])
    print(fig.data[0].hovertemplate, np.asarray(fig.data[0].customdata)[:1])


@probe
def px_wide_index():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=pd.Index([10, 20, 30], name="idx"))
    fig = px.bar(df, orientation="h")
    print(fig.data[0].x, fig.data[0].y, fig.layout.yaxis.title.text)
    fig = px.line(df, facet_col="variable")
    print([t.xaxis for t in fig.data])
