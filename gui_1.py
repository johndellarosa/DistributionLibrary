import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import numpy as np
import pandas as pd
import JohnDistribution as jd

# Read in the data
# data = pd.read_csv("precious_metals_prices_2018_2021.csv")
# data['DateTime'] = pd.to_datetime(data['DateTime'])

dist = jd.Gaussian(0,1)

min_x = -5
max_x = 5
interval = .01

x = np.arange(min_x, max_x + interval, interval)
y = np.array([dist.PDF(i) for i in x])
data = pd.DataFrame(index=x)

data['y'] = y
#print(data)

fig = px.line(
    data,
    title="Gaussian",
    x=data.index,
    y=y,
    color_discrete_map={"Gold": "gold"}
)

app = dash.Dash(__name__)
app.title = "Distribution GUI"

app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id="header-area",
            children=[
                html.H1(
                    id="header-title",
                    children="Distribution Graph",

                ),
                # html.P(
                #     id="header-description",
                #     children=("The cost of precious metals", html.Br(), "between 2018 and 2021"),
                # ),
            ],
        ),
        html.Div(
            id="menu-area",
            children=[
                html.Div(
                    children=[
                        # html.Div(
                        #     className="menu-title",
                        #     children="Metal"
                        # ),
                        # dcc.Dropdown(
                        #     id="metal-filter",
                        #     className="dropdown",
                        #     options=[{"label": metal, "value": metal} for metal in data.columns[1:]],
                        #     clearable=False,
                        #     value="Gold"
                        # )
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            className="menu-title",
                            children="Mean"
                        ),
                        # dcc.DatePickerRange(
                        #     id='date-range',
                        #     min_date_allowed=data.DateTime.min().date(),
                        #     max_date_allowed=data.DateTime.max().date(),
                        #     start_date=data.DateTime.min().date(),
                        #     end_date=data.DateTime.min().date()
                        #
                        # )
                        dcc.Input(id="input1", type="text", placeholder='0', style={'marginRight': '10px'},value='0'),
                        html.Div(
                            className='menu-title',
                            children='Variance'
                        ),
                        dcc.Input(id="input2", type="text", placeholder='1', style={'marginRight': '10px'},value='1'),
                    ]
                )
            ]
        ),
        html.Div(
            id="graph-container",
            children=dcc.Graph(
                id="price-chart",
                figure=fig,
                config={"displayModeBar": False}
            ),
        ),
    ]
)

@app.callback(
    Output("price-chart", "figure"),
    Input("input1", "value"),
    Input('input2','value')
)
def update_chart(value1,value2):
    try:
        value1 = float(value1)
        value2 = float(value2)
        print("success")
    except:
        print(value1)
        print(value2)
        raise dash.exceptions.PreventUpdate

    # filtered_data = data.loc[(data.DateTime >= start_date) & (data.DateTime <= end_date)]

    dist = jd.Gaussian(value1, value2)

    min_x = value1-2*value2
    max_x = value1+2*value2
    interval = .01

    x = np.arange(min_x, max_x + interval, interval)
    y = np.array([dist.PDF(i) for i in x])
    data = pd.DataFrame(index=x)

    data['y'] = y
    # print(data)

    fig = px.line(
        data,
        title="Gaussian",
        x=data.index,
        y=y,
        color_discrete_map={"Gold": "gold"}
    )

    # Create a plotly plot for use by dcc.Graph().
    # fig = px.line(
    #     filtered_data,
    #     title="Precious Metal Prices 2018-2021",
    #     x="DateTime",
    #     y=[metal],
    #     color_discrete_map={
    #         "Platinum": "#E5E4E2",
    #         "Gold": "gold",
    #         "Silver": "silver",
    #         "Palladium": "#CED0DD",
    #         "Rhodium": "#E2E7E1",
    #         "Iridium": "#3D3C3A",
    #         "Ruthenium": "#C9CBC8"
    #     }
    # )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="x",
        yaxis_title="f(x)",
        font=dict(
            family="Verdana, sans-serif",
            size=18,
            color="white"
        ),
    )
    print(value1)
    print(value2)
    print('returning fig')
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
