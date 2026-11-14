import json, traceback, warnings
import numpy as np
import pandas as pd
import polars as pl
import plotly
import orjson
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")
print("plotly", plotly.__version__, "orjson", orjson.__version__, "default engine", pio.json.config.default_engine)


def show(label, fig):
    out = []
    for eng in ["json", "orjson"]:
        try:
            d = json.loads(pio.to_json(fig, engine=eng))
            out.append(json.dumps(d["data"][0].get("x"))[:160])
        except Exception as e:
            out.append("ERR " + type(e).__name__ + ": " + str(e)[:120])
    print(f"{label}\n   json  : {out[0]}\n   orjson: {out[1]}")


dt = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[ns]")
show("go numpy datetime64[ns] w/ NaT", go.Figure(go.Scatter(x=dt, y=[1, 2, 3])))
show("go numpy datetime64[D] w/ NaT", go.Figure(go.Scatter(x=dt.astype("datetime64[D]"), y=[1, 2, 3])))
show("go pd.Series datetime w/ NaT", go.Figure(go.Scatter(x=pd.Series(dt), y=[1, 2, 3])))
show("go DatetimeIndex w/ NaT", go.Figure(go.Scatter(x=pd.DatetimeIndex(dt), y=[1, 2, 3])))
show("go tz Series w/ NaT", go.Figure(go.Scatter(x=pd.Series(dt).dt.tz_localize("UTC"), y=[1, 2, 3])))
df = pd.DataFrame({"t": dt, "y": [1, 2, 3]})
show("px pandas datetime w/ NaT", px.scatter(df, x="t", y="y"))
show("px pandas tz datetime w/ NaT", px.scatter(df.assign(t=df.t.dt.tz_localize("Europe/Paris")), x="t", y="y"))
show("px polars datetime w/ null", px.scatter(pl.from_pandas(df), x="t", y="y"))
show("px line pandas datetime w/ NaT", px.line(df, x="t", y="y"))

# what actually lands in the HTML (default engine auto -> orjson)
fig = go.Figure(go.Scatter(x=dt, y=[1, 2, 3]))
html = pio.to_html(fig, include_plotlyjs=False)
i = html.find('"x":')
print("to_html snippet:", html[i:i + 120])
print("fig.to_json default:", fig.to_json()[:120])

# ---- size NaN
try:
    px.scatter(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "s": [1.0, np.nan, 3.0]}), x="x", y="y", size="s")
    print("size nan ok")
except Exception as e:
    print("size nan ERR", str(e).strip().splitlines()[0])
