from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import HoverTool, CrosshairTool, ColumnDataSource, LinearColorMapper, Scatter
from bokeh.palettes import tol as sp
import numpy as np

def plot(df_values, df_results, pred=0):
    date_dict = {"date": df_values['timestamp']}
    res_dict = {col: df_results[col] for col in df_results.columns}
    val_dict = {col: df_values[col] for col in df_values.columns}
    pred_dict = {"pred": pred}
    data = res_dict | val_dict
    data = data | date_dict
    data = data | pred_dict
    data = ColumnDataSource(data)

    tool_tips_top = [
        ("Time", "@date"),
        ("Value", "$y")]


    top = figure(height=300, width=1000,
                 #x_axis_type="datetime",
                 x_axis_location="above",
                 background_fill_color="beige", tooltips=tool_tips_top,
                 title="Data_output")

    tool_tips_top = [
        ("Time", "@date_print"),
        ("Value", "$y")]
    """
    top.line(x="date", y="heart_rate", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="enhanced_speed", source=data, line_color="green", line_width=2)
    top.line(x="date", y="distance", source=data, line_color="blue", line_width=2)
    #top.line(x="date", y="position_long", source=data, line_color="red", line_width=2)
    """

    top.line(x="date", y="channel_12", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="channel_13", source=data, line_color="green", line_width=2)
    top.line(x="date", y="channel_14", source=data, line_color="blue", line_width=2)
    top.line(x="date", y="channel_15", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_16", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_17", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_18", source=data, line_color="red", line_width=2)
    top.line(x="date", y="channel_19", source=data, line_color="red", line_width=2)
    top.scatter(x="date", y="pred", source=data, size=2)
    top.scatter(x="date", y="anom", source=data, color="red", size=2)
    #top.grid.grid_line_width = 2
    top.add_tools(HoverTool(mode='vline', tooltips=tool_tips_top))
    top.add_tools(CrosshairTool(dimensions="height"))

    #Make bottom figure
    bottom = figure(height=300, width=1000,
                    x_axis_type="datetime",
                    x_axis_location="below",
                    background_fill_color="beige",
                    title="Outlier Scores")

    tool_tips_bottom = [
        ("Time", "@date_print"),
        ("Score Fourier", "@out0"),
        ("Score Signature", "@out1"),
        ("Score no transform", "@out2")
    ]


    bottom.add_tools(HoverTool(mode='vline', tooltips=tool_tips_bottom))
    bottom.add_tools(CrosshairTool(dimensions="height"))

    bottom.line(x="date", y="out1",legend_label="Distance", source=data, line_color="blue", line_width=1)
    #bottom.line(x="date", y="out2", source=data, line_color="red", line_width=1)
    bottom.line(x="date", y="out3",legend_label="Signature", source=data, line_color="green", line_width=1)


    show(column(top, bottom))



