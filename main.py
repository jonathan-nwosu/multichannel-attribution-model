# 1) Import packages needed to perform tasks

import numpy as np
import pandas as pd
from ChannelAttribution import *
import plotly.io as pio

# 2) Read file

Data = pd.read_csv("data.csv",sep=";")

# 3) Load heuristic model (shows first click, last click and linear attribution models)

H = heuristic_models(Data,"path","total_conversions",var_value="total_conversion_value")

# 4) Load Markov model (with relevant paramters)

M = markov_model(Data, "path", "total_conversions", var_value="total_conversion_value")

# 5) Load Markov model

R = pd.merge(H,M,on="channel_name",how="inner")
R1=R[["channel_name","first_touch_conversions","last_touch_conversions",\
      "linear_touch_conversions","total_conversions"]]

# 6) Load Markov model
R1.columns = ["channel_name","first_touch","last_touch","linear_touch","markov_model"]

R1 = pd.melt(R1, id_vars="channel_name")

# 7) Ploting data on histogram

data = [dict(
    type = "histogram",
    histfunc="sum",
    x = R1.channel_name,
    y = R1.value,
    transforms = [dict(
    type = "groupby",
    groups = R1.variable,
    )],
)]

#8) Show and open graph

fig = dict({"data":data})
pio.show(fig,validate=False)
