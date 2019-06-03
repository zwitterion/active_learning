import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

app = dash.Dash(__name__)

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
df = pd.read_csv('./iris.csv')
#labels = df["label"].unique()
labels = ["a", "b"]

df["label"] = " "

columns = [{"name": i, "id": i, "editable": False}  for i in df.columns if (i != "label")]
columns.insert(0,{"name": "label", "id": "label", "editable": True, "presentation":"dropdown"})


app.layout = html.Div([
    dash_table.DataTable(
        id='data-table',
        columns=columns,
        data=df.to_dict('records'),
        editable=True,
        column_static_dropdown=[
            {
                'id': 'label',
                'dropdown': [
                    {'label': i, 'value': i} for i in labels
                ]
            }
        ]
    ),
    html.Div(id='table-dropdown-container')
])

# In order for the changes in the dropdown to persist,
# the dropdown needs to be "connected" to the table via
# a callback
@app.callback(Output('table-dropdown-container', 'children'),
              [Input('data-table', 'data_timestamp')])
def update_output(timestamp):
    return timestamp

"""
@app.callback(
    Output('computed-table', 'data'),
    [Input('computed-table', 'data_timestamp')],
    [State('computed-table', 'data')])
def update_columns(timestamp, rows):
    print(rows)
    for row in rows:
        try:
            row['output-data'] = float(row['input-data']) ** 2
        except:
            row['output-data'] = 'NA'
    return rows
"""

if __name__ == '__main__':
    app.run_server(debug=True)