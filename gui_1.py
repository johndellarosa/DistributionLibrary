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

continuous_distribution_dict = {
    'Normal':jd.Gaussian,
    'Cauchy':jd.Cauchy,
    'Exponential':jd.Exponential,
    'Erlang':jd.Erlang,
    'Gamma':jd.Gamma,
    'Inverse Gamma':jd.Inverse_Gamma,
    'Log Normal':jd.Log_Normal,
    'Levy':jd.Levy,
    'Laplace':jd.Laplace,
    'Pareto':jd.Pareto,
}

param_requirements = {
    'Normal':'''
        Param 1 (mu): float (-infty, infty)
        Param 2 (sigma): float (0, infty)
        ''',
    'Cauchy':'''
        Param 1 (Gamma): float (0, infty)
        Param 2 (x_0): float (-infty, infty)
        ''',
    'Exponential':'''
        Param 1 (Beta): float (0, infty)
        ''',
    'Erlang':'''
        Param 1 (k): int [1,infty)
        Param 2 (lambda): float (0, infty)
        ''',
    'Gamma':'''
        Param 1 (alpha): float (0, infty)
        Param 2 (beta): float (0, infty)
        ''',
    'Inverse Gamma':'''
        Param 1 (alpha): float (0, infty)
        Param 2 (beta): float (0, infty)
        ''',
    'Log Normal':'''
        Param 1 (mu): float (-infty, infty)
        Param 2 (sigma): float (0, infty)
        ''',
    'Levy':'''
        Param 1 (mu): float (-infnty, infty)
        Param 2 (c): float (0, infty)
        ''',
    'Laplace':'''
        Param 1 (mu): float (-infty, infty)
        Param 2 (b): float (0, infty)
        ''',
    'Pareto':'''
        Param 1 (x_m): float (0, infty)
        Param 2 (alpha): float (0, infty)
        '''
}


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
                        html.Div(
                            className="menu-title",
                            children="Distribution"
                        ),
                        dcc.Dropdown(
                            id="dist-basis",
                            className="dropdown",
                            options=[{"label": dist, "value": dist} for dist in continuous_distribution_dict.keys()],
                            clearable=False,
                            value="Normal"
                        )
                    ]
                ),
                html.Div(
                    children=[

                        html.Div(id='param-restrictions',children=''),
                        html.Div(
                            className="menu-title",
                            children=["Param 1",
dcc.Input(id="input1", type="text", placeholder='0', style={'marginRight': '10px'},value='0'),
                                      ]
                        ),
                        # dcc.DatePickerRange(
                        #     id='date-range',
                        #     min_date_allowed=data.DateTime.min().date(),
                        #     max_date_allowed=data.DateTime.max().date(),
                        #     start_date=data.DateTime.min().date(),
                        #     end_date=data.DateTime.min().date()
                        #
                        # )

                        html.Div(
                            className='menu-title',
                            children=['Param 2',
                                      dcc.Input(id="input2", type="text", placeholder='1',
                                                style={'marginRight': '10px'}, value='1'),

                                      ]
                        ),
                        html.Div(
                            className='menu-title',
                            children=['Min x',
                                      dcc.Input(id="minX", type="text", placeholder='-5',
                                                style={'marginRight': '10px'}, value='-5'),

                                      ]
                        ),
                        html.Div(
                            className='menu-title',
                            children=['Max x',
                                      dcc.Input(id="maxX", type="text", placeholder='5',
                                                style={'marginRight': '10px'}, value='5'),

                                      ]
                        ),

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
    Output('param-restrictions','children'),
    Output("price-chart", "figure"),
    Input('dist-basis','value'),
    Input("input1", "value"),
    Input('input2','value'),
    Input('minX','value'),
    Input('maxX','value')
)
def update_chart(dist_type,value1,value2,minX,maxX):
    try:
        value1 = float(value1)
        value2 = float(value2)
        minX = float(minX)
        maxX = float(maxX)
        print("success")
    except:
        print(value1)
        print(value2)
        raise dash.exceptions.PreventUpdate

    # filtered_data = data.loc[(data.DateTime >= start_date) & (data.DateTime <= end_date)]

    try:
        dist = continuous_distribution_dict[dist_type](value1, value2)


        interval = .01

        x = np.arange(minX, maxX, interval)
        y = np.array([dist.PDF(i) for i in x])
        data = pd.DataFrame(index=x)

        data['y'] = y
        # print(data)

        fig = px.line(
            data,
            title=f"{dist_type}",
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
            template="seaborn",
            xaxis_title="x",
            yaxis_title="f(x)",
            font=dict(
                family="Verdana, sans-serif",
                size=18,
                color="black"
            ),
        )
        print(value1)
        print(value2)

        print(param_requirements[dist_type])
        print('returning fig')
        return param_requirements[dist_type] ,fig

    except Exception as e:
        print('error')
        print(e)
        #raise dash.exceptions.PreventUpdate
        fig = px.line(

            title=f"{dist_type}",
            x=0,
            y=0,
            color_discrete_map={"Gold": "gold"}
        )
        return param_requirements[dist_type], fig


if __name__ == "__main__":
    app.run_server(debug=True)
