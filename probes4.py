import numpy as np
import plotly
import plotly.express as px

print("plotly", plotly.__version__)
df = px.data.tips()


def doms(fig, letter):
    return [tuple(round(v, 4) for v in fig.layout[k].domain) for k in sorted(
        (k for k in fig.layout if k.startswith(letter + "axis")), key=lambda k: int(k[5:] or 1))]


for sp in [None, 0, 0.0, 1e-9, 0.05]:
    fig = px.scatter(df, x="total_bill", y="tip", facet_row="sex", facet_col="day",
                     facet_row_spacing=sp, facet_col_spacing=sp)
    print(f"scatter facet_row/col_spacing={sp!r}: y domains {doms(fig, 'y')[:2]} | x domains {doms(fig, 'x')[:2]}")

fig = px.scatter(df, x="total_bill", y="tip", facet_col="day", facet_col_wrap=2, facet_row_spacing=0)
print("facet_col_wrap=2, facet_row_spacing=0: y domains", doms(fig, "y")[::2])

from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2, horizontal_spacing=0, vertical_spacing=0)
print("make_subplots spacing=0 (reference): x domains", doms(fig, "x")[:2], "y domains", doms(fig, "y")[::2])

img = np.random.default_rng(0).random((2, 4, 4))
fig = px.imshow(img, facet_col=0, facet_col_spacing=0)
print("imshow facet_col_spacing=0: x domains", doms(fig, "x"))
