import numpy as np
import pandas as pd
import ipywidgets
from plotly.subplots import make_subplots
import plotly.graph_objects as pgo
import math
import sympy as sym

def dmb_dt(mb: float, bm: float, appeal_bm: float) -> float:
    """feelings of Mark towards bridget"""
    #return f1_return(mb) + f2_return(bm) + f2_return(appeal_b)
    return ((1 + synergy(mb, 0.5)) * f2_return(bm)) + ((1 + synergy(mb, 0.5))*f2_return(appeal_bm))


def dmb_dt2(mb: float, bm: float, appeal_bm: float, time: float) -> float:
    """feelings of Mark towards bridget WITH DYNAMIC APPEAL"""
    #return f1_return(mb) + f2_return(bm) + f2_return(appeal_b)
    return ((1 + synergy(mb, 0.5)) * f2_return(bm)) + ((1 + synergy(mb, 1))*f2_return(sym.diff(appeal(time)) + appeal_bm))


def ddb_dt(db: float, bd: float, appeal_bd: float) -> float:
    """feelings of Daniel towards Bridget"""
    #return f1_return(db) + f1_return(bd) + f1_return(appeal_b)
    return f1_return(bd) + f2_return(appeal_bd)


def dbm_dt(bm: float, mb: float, appeal_m: float, beta: float, bd: float) -> float:
    """feelings of Bridget towards Mark"""
    #return f2_return(bm) + f1_return(mb) + f1_return(appeal_m) - beta * bd
    return ((1 + synergy(bm, 0.5)) * f1_return(mb)) + ((1 + synergy(bm, 0.5)) * f2_return(appeal_m)) - beta * bd


def dbd_dt(bd: float, db: float, appeal_d: float, beta: float, bm: float) -> float:
    """feelings of Bridget towards Daniel"""
    #return f2_return(bd) + f1_return(db) + f1_return(appeal_d) - beta * bm
    return ((1 + synergy(bd, 0.5)) * f1_return(db)) + ((1 + synergy(bm, 0.5)) * f2_return(appeal_d)) - beta * bm


def f1_return(x: float) -> float:
    if x <= 0:
        return (2 * x)/(1 - x)
    else:
        return (x / (2 + x)) * (10**6 - x ** 8) / (10**6 + x ** 8)


def f2_return(x: float) -> float:
    f_plus = 0.5
    f_minus = -0.5
    if x <= 0:
        return (f_minus * x) / (x - 1)
    else:
        return (f_plus * x) / (x + 1)


def synergy(x: float, s: float) -> float:
    """Individual reactions being enhanced by love"""
    if x < 0:
        return 0
    else:
        return (s * (x**8)) / (1 + x**8)

def appeal(x: float) -> float:
    if x < 6:
        return 0.5/ (1 + math.e**(-2 * (x - 2)))
    else:
        return 0


def solver(time: float, mb_0: float, db_0: float, bm_0: float, bd_0: float,
           appeal_bm: float, appeal_bd: float, appeal_m: float, appeal_d: float, beta: float):
    """solves the love triangle over some time given initial states"""
    dt = 0.04
    steps = int(time // dt)
    mb_array = np.zeros(shape=steps + 1)
    mb_array[0] = mb_0
    db_array = np.zeros(shape=steps + 1)
    db_array[0] = db_0
    bm_array = np.zeros(shape=steps + 1)
    bm_array[0] = bm_0
    bd_array = np.zeros(shape=steps + 1)
    bd_array[0] = bd_0
    for i in range(0, steps):
        dmb = dmb_dt(mb=mb_array[i], bm=bm_array[i], appeal_bm=appeal_bm) * dt
        mb_array[i + 1] = mb_array[i] + dmb
        ddb = ddb_dt(db=db_array[i], bd=bd_array[i], appeal_bd=appeal_bd) * dt
        db_array[i + 1] = db_array[i] + ddb
        dbm = dbm_dt(bm=bm_array[i], mb=mb_array[i], appeal_m=appeal_m, beta=beta, bd=bd_array[i]) * dt
        bm_array[i + 1] = bm_array[i] + dbm
        dbd = dbd_dt(bd=bd_array[i], db=db_array[i], appeal_d=appeal_d, beta=beta, bm=bm_array[i]) * dt
        bd_array[i + 1] = bd_array[i] + dbd
    return pd.DataFrame(
        {
            "time": np.arange(0, time, dt),
            "Mark-Bridget": mb_array,
            "Daniel-Bridget": db_array,
            "Bridget-Mark": bm_array,
            "Bridget-Daniel": bd_array,
        }
    )


class PhasePortrait:
    """creates an interactive phase portrait"""
    def __init__(self):
        self.total_time_weeks = 12

        self.mark_bridget_initial = ipywidgets.FloatSlider(min=-15.0, max=15.0)
        self.mark_bridget_initial.observe(self.update_plot, names="value")

        self.daniel_bridget_initial = ipywidgets.FloatSlider(min=-15.0, max=15.0)
        self.daniel_bridget_initial.observe(self.update_plot, names="value")

        self.bridget_mark_initial = ipywidgets.FloatSlider(min=-15.0, max=15.0)
        self.bridget_mark_initial.observe(self.update_plot, names="value")

        self.bridget_daniel_initial = ipywidgets.FloatSlider(min=-15.0, max=15.0)
        self.bridget_daniel_initial.observe(self.update_plot, names="value")

        self.bridget_appeald = ipywidgets.FloatSlider(min=-2.0, max=2.0)
        self.bridget_appeald.observe(self.update_plot, names="value")
        
        self.bridget_appealm = ipywidgets.FloatSlider(min=-2.0, max=2.0)
        self.bridget_appealm.observe(self.update_plot, names="value")

        self.mark_appeal = ipywidgets.FloatSlider(min=-2.0, max=2.0)
        self.mark_appeal.observe(self.update_plot, names="value")

        self.daniel_appeal = ipywidgets.FloatSlider(min=-2.0, max=2.0)
        self.daniel_appeal.observe(self.update_plot, names="value")

        self.beta = ipywidgets.FloatSlider(min=-1.0, max=1.0)
        self.beta.observe(self.update_plot, names="value")

        self.phase_plot_fig = pgo.FigureWidget(make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=("Mark & Bridget", "Daniel & Bridget", "Feelings Over Time")
        ))
        self.phase_plot_fig.update_xaxes(row=1, col=1, title_text="MB")
        self.phase_plot_fig.update_yaxes(row=1, col=1, title_text="BM")
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Mark & Bridget"), row=1, col=1
        )
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Daniel & Bridget"), row=1, col=2
        )
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Mark towards Bridget"), row=2, col=1
        )
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Daniel towards Bridget"), row=2, col=1
        )
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Bridget towards Mark"), row=2, col=1
        )
        self.phase_plot_fig.add_trace(
            pgo.Scatter(x=[], y=[], mode="lines", name="Bridget towards Daniel"), row=2, col=1
        )
        self.widget = ipywidgets.VBox([
            self.phase_plot_fig,
            ipywidgets.HBox([
                ipywidgets.Label("Mark's initial feelings towards Bridget"),
                self.mark_bridget_initial,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("Daniel's initial feelings towards Bridget"),
                self.daniel_bridget_initial,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("Bridget's initial feelings towards Mark"),
                self.bridget_mark_initial,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("Bridget's initial feelings towards Daniel"),
                self.bridget_daniel_initial,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("Bridget's appeal to Daniel"),
                self.bridget_appeald,
            ]),
            
             ipywidgets.HBox([
                ipywidgets.Label("Bridget's appeal to Mark"),
                self.bridget_appealm,
            ]),
            
            ipywidgets.HBox([
                ipywidgets.Label("Mark's appeal"),
                self.mark_appeal,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("Daniel's appeal"),
                self.daniel_appeal,
            ]),
            ipywidgets.HBox([
                ipywidgets.Label("beta"),
                self.beta,
            ]),
        ])

    def update_plot(self, change: dict):
        """updates the figure widget"""
        love_df = solver(
            time=self.total_time_weeks,
            mb_0=self.mark_bridget_initial.value,
            db_0=self.daniel_bridget_initial.value,
            bm_0=self.bridget_mark_initial.value,
            bd_0=self.bridget_daniel_initial.value,
            appeal_d=self.daniel_appeal.value,
            appeal_m=self.mark_appeal.value,
            appeal_bm=self.bridget_appealm.value,
            appeal_bd=self.bridget_appeald.value,
            beta=self.beta.value
        )
        mark_bridget_phase = self.phase_plot_fig.data[0]
        mark_bridget_phase.x = love_df["Mark-Bridget"].to_numpy()
        mark_bridget_phase.y = love_df["Bridget-Mark"].to_numpy()

        daniel_bridget_phase = self.phase_plot_fig.data[1]
        daniel_bridget_phase.x = love_df["Daniel-Bridget"].to_numpy()
        daniel_bridget_phase.y = love_df["Bridget-Daniel"].to_numpy()

        mark_towards_bridget = self.phase_plot_fig.data[2]
        mark_towards_bridget.x = love_df["time"].to_numpy()
        mark_towards_bridget.y = love_df["Mark-Bridget"].to_numpy()

        daniel_towards_bridget = self.phase_plot_fig.data[3]
        daniel_towards_bridget.x = love_df["time"].to_numpy()
        daniel_towards_bridget.y = love_df["Daniel-Bridget"].to_numpy()

        bridget_towards_mark = self.phase_plot_fig.data[4]
        bridget_towards_mark.x = love_df["time"].to_numpy()
        bridget_towards_mark.y = love_df["Bridget-Mark"].to_numpy()

        bridget_towards_daniel = self.phase_plot_fig.data[5]
        bridget_towards_daniel.x = love_df["time"].to_numpy()
        bridget_towards_daniel.y = love_df["Bridget-Daniel"].to_numpy()










