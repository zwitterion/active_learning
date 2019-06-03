#
# css: https://codepen.io/chriddyp/pen/bWLwgP
#
import pandas as pd
import numpy as np
import json

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table

from active_learner import ActiveLearner
import argparse

ui_label_accuracy = "Accuracy"
ui_string_total_labeled = "You have labeled {:6.2f} % of {} rows."

parser = argparse.ArgumentParser()
parser.add_argument("--port", help="port number",
                    type=int, default=80)
parser.add_argument("--dataset", help="test dataset 0: iris 1: prev. maintenance",
                    type=int, default=0)


args = parser.parse_args()

def generate_labels_table(dataframe, percent, total_rows, max_rows=10):
    
    title = html.Div(children=ui_string_total_labeled.format(percent, total_rows))
    table = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ],id="r{}".format(i)) for i in range(min(len(dataframe), max_rows))]
    ,style={"margin":"auto"})
    return html.Div(children=[title,table], style={"text-align": "center"})

def display_labels_pie_chart(labels):
    

    unique_labels, counts = np.unique(labels, return_counts=True)
    
    
    print("COUNTS", counts)
    
    data = [
        {
            'labels': unique_labels,
            'values': counts,
            'type': 'pie',
        },
    ]

    return html.Div([
        dcc.Graph(
            id='graph',
            figure={
                'data': data,
                'layout': {
                    'title': 'Predictions by Label',
                    'margin': {
                        'l': 5,
                        'r': 5,
                        'b': 50,
                        't': 50
                    },
                    'legend': {'x': 0, 'y': 1}
                }
            },
            style={"border": "0px solid #DCDCDC", "width": "65%"}
        )
        ], 
        #className="three columns"
        )
     



def generate_data_table(dataframe, columns, labels):
    return dash_table.DataTable(
        id='table-dropdown',
        pagination_mode="fe",
        pagination_settings={
                "displayed_pages": 1,
                "current_page": 0,
                "page_size": 10,
        },
        navigation="page",
        filtering=True,
        sorting=True,
        sorting_type="single",
        data=dataframe.to_dict('records'),
        columns=columns,
        editable=True,
        column_static_dropdown=[
            {
                'id': 'label',
                'dropdown': [
                    {'label': i, 'value': i}
                    for i in labels
                ]
            }
        ]
    )

# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/find-sample-size/
#def get_sample_size():
    

app = dash.Dash(__name__)
df = None

if args.dataset==0:
    df = pd.read_csv('./iris.csv')
    labels = df["label"].unique()

    # df["debug"] = df["label"]
    df["label"] = ""
    df["predicted"] = ""
    df["entropy"] = 0
else:
    df = pd.read_csv('./iris.csv')


columns = [{"name": i, "id": i, "editable": False}  for i in df.columns if (i not in ["label", "predicted"])]
columns.insert(0,{"name": "label", "id": "label", "editable": True, "presentation":"dropdown"})
columns.insert(1,{"name": "predicted", "id": "predicted", "editable": False})


html.Div([
    html.Div('Example Div', style={'color': 'blue', 'fontSize': 14}),
    html.P('Example P', className='my-class', id='my-p-element')
], style={'marginBottom': 50, 'marginTop': 25})

app.layout = html.Div([
    html.Div([
        html.Label('Labels', htmlFor='labels-list'),
        dcc.Input(id='labels-list', value='yes,no', type='text'),
    ], style={'marginBottom': 50, 'marginTop': 25}),
    html.Div(id='table-dropdown-container', children=generate_data_table(df, columns, ['x', 'y'])),
    html.Div(id='stats', children=[
        html.Div(children=[
            html.Label(ui_label_accuracy, htmlFor="stats-score"),
            html.Div(id='stats-score'),
            html.Div(["* Evaluated on ", html.Span(id="stats-eval-size",children="0"), " rows"])
            ],
            className="two columns", style={"margin": "20px", "padding":"20px", "border":"1px solid #DADADA", "text-align": "center"}),
        html.Div(id='stats-class-counts', className="two columns"),
        html.Div(id='labels-pie-chart', className="three columns")
    ]),    
])


@app.callback(Output('table-dropdown-container', 'children'),
              [Input('labels-list', 'value')]
              )
def update_dropdown(data):
    labels = data.split(",")
    
    
    return generate_data_table(df, columns, labels)


@app.callback([Output('table-dropdown', 'data'),
              Output('stats-score', 'children'),
              Output('stats-eval-size', 'children'),
              Output('stats-class-counts', 'children'),
              Output('labels-pie-chart', 'children')],
              [Input('table-dropdown', 'data_timestamp')],
              [State('table-dropdown', 'data'), 
              State('labels-list', 'value')]
              
              )
def update_output(timestamp, rows, labels):
    

    df = pd.DataFrame(rows)    
    
    X = df.dropna()
    X = X[X["label"]!=""]
    
    y = X["label"]

    columns = [c for c in df.columns if c not in ["label", "predicted", "debug"]]
    X = X[columns]

    print("X: ", len(X), "Y:", len(y))

    # class counts on label data set
    #     
    percent = 100.*(len(y)/len(df))
    class_counts_summary = generate_labels_table(y.to_frame().groupby("label").size().to_frame("count").reset_index(), percent, len(df))

    if (len(X)> 3):

        if (len(y.unique())) >= 2: # need at least two classes


            al = ActiveLearner(X, y)
            if al.train():
                print(al.score)
                print("XTEST", len(al.X_test))

                y_hat = al.model.predict(df[columns])
                proba = al.model.predict_proba(df[columns])

                for i in range(0, len(rows)):
                    rows[i]["predicted"] = y_hat[i]
                    rows[i]["entropy"] = al.entropy(proba[i])


                return [rows, "{:6.2f}".format(al.score), len(al.X_test), class_counts_summary, display_labels_pie_chart(y_hat)]
            else:
                print("unable to train - not enough samples")
    


    #return [rows, 0, class_counts, display_labels_pie_chart([1,2,3])]
    return [rows, 0, 0, class_counts_summary, ""]


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0",port=args.port)