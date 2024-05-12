import dash
import time
from dash import html, dcc, dash_table, no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.io as pio
import pandas as pd
from AnalysisScript import *
"""Need to note that in order for images to export properly you have to install kaleido. pip install -U kaleido"""
#Logan Buddenbaum certified Code

# Base class for buttons
class CustomButton:
    def __init__(self, label):
        self.label = label

    def layout(self):
        return html.Button(self.label, id=self.label.replace(" ", "-").lower() + '-button')

    def callback(self, app, df):
        pass

# Derived class for downloading graphs
class DownloadButton(CustomButton):
    def __init__(self, app, label, filename):
        super().__init__(label)
        self.app = app
        self.filename = filename
        self.button_id = f"{label.replace(' ', '-').lower()}-button"
        self.download_id = f"{label.replace(' ', '-').lower()}-download"
        self.register_callback()

    def layout(self):
        return html.Div([
            html.Button(self.label, id=self.button_id),
            dcc.Download(id=self.download_id)
        ])

    def register_callback(self):
        @self.app.callback(
            Output(self.download_id, 'data'),
            [Input(self.button_id, "n_clicks")],
            [State('figure-store', 'data'),
             State('figure-select-dropdown', 'value')]
            
        )
        def trigger_download(n_clicks, figures, selected_figure_id):
            if n_clicks and figures and selected_figure_id:
                figure_data = figures.get(selected_figure_id)
                if figure_data:
                    return dcc.send_bytes(
                        lambda buffer: buffer.write(pio.to_image(figure_data, format='png')),
                        f"{selected_figure_id}.png"
                        )

# Load data
df = pd.read_csv('realtor-data.zip.csv')
dfNoMissingValues = remove_rows_with_missing_values(df)
dfNoMissingValues1, dfOutliers = remove_outliers(dfNoMissingValues, 'price')
dfNoMissingValues2, df2Outliers = remove_outliers(dfNoMissingValues1, 'bed')
dfNoMissingValues3, df3Outliers = remove_outliers(dfNoMissingValues2, 'bath')
dfNoMissingValues4, df4Outliers = remove_outliers(dfNoMissingValues3, 'acre_lot')
dfNoMissingValues5, df5Outliers = remove_outliers(dfNoMissingValues4, 'house_size')
include_columns = ['price','bed', 'bath', 'acre_lot', 'house_size']
dfcleaneddates = prepare_date_features(dfNoMissingValues5, 'prev_sold_date')
X_train, X_test, Y_train, Y_test = prepare_and_split_data(dfcleaneddates)
columns_to_encode = ['state', 'city', 'zip_code']
X_train, X_test = apply_multi_level_target_encoding(X_train, X_test, Y_train, columns_to_encode)
X_train, X_test = apply_cyclical_encoding(X_train, X_test, 'day', 'month')

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Tabs for different graphs or models
download_button = DownloadButton(app, "Download Selected Graph", "selected-graph.png")
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Summary Functions of DataFrame', value='tab-summary'),
        dcc.Tab(label='Bar Chart that shows columns with missing data', value='tab-missingdata'),
        dcc.Tab(label='Skewness and Kurtosis', value='tab-skku'),
        dcc.Tab(label='Histograms', value='tab-1'),
        dcc.Tab(label='Bar Charts', value='tab-2'),
        dcc.Tab(label='Correlation Matrix', value='tab-3'),
        dcc.Tab(label='Price Trends Over Time', value='tab-4'),
        dcc.Tab(label='Location Price Effects', value='tab-5'),
        dcc.Tab(label='Property Sizes Versus Price Trends', value='tab-6'),
        dcc.Tab(label='Scatterplots', value='tab-7'),
        dcc.Tab(label='Linear Regression Model', value='linear-regression'),
        dcc.Tab(label='Random Forest Model', value='random-forest'),
        dcc.Tab(label='Gradient Boosting Model', value='gradient-boosting')
    ]),
    html.Div(id='tabs-content'),
    dcc.Store(id='figure-store'),
    dcc.Dropdown(id='figure-select-dropdown'),
    download_button.layout()
])

# Callback for tabs content
@app.callback([Output('tabs-content', 'children'),
               Output('figure-store', 'data'),
               Output('figure-select-dropdown', 'options'),
               Output('figure-select-dropdown', 'value')],
              [Input('tabs', 'value')]
              )
def render_content(tab):
    if tab == 'tab-summary':
        head_df, describe_df = summarystatsbymethods(dfcleaneddates)
        content = html.Div([
            html.H3("Head of the DataFrame"),
            dash_table.DataTable(
                data=head_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in head_df.columns],
                style_table={'overflowX':'scroll'},
                style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px'}
            ),
            html.H3("Descriptive Statistics"),
            dash_table.DataTable(
                data=describe_df.to_dict('records'),
                columns=[{'name': 'index', 'id': 'index'}] + [{'name': str(col), 'id': str(col)} for col in describe_df.columns if col != 'index'],
                style_table={'overflowX':'scroll'},
                style_cell={'textAlign': 'left', 'minWidth': '150px', 'width': '150px', 'maxWidth': '150px'}
            )
        ])
        return content, {}, [], None
    elif tab == 'tab-missingdata':
        fig = missingvalsVisuals(df)
        content = html.Div([
            dcc.Graph(id='missingdata', figure=fig)
        ])
        figures = {'fig1': fig}
        dropdown_options = [{'label': 'Bar Chart of Missing Data', 'value': 'fig1'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-skku':
        skewness, kurtosis = skewkurtfunction(dfcleaneddates)
        content = html.Div([
            html.H3("Skewness"),
            dash_table.DataTable(
                data=skewness.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in skewness.columns],
                style_table={'overflowX': 'auto'}
            ),
            html.H3("Kurtosis"),
            dash_table.DataTable(
                data=kurtosis.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in kurtosis.columns],
                style_table={'overflowX': 'auto'}
            )
        ])
        return content, {}, [], None
    elif tab == 'tab-1':
        fig1 = plot_histogramv2(dfNoMissingValues5, 'price', bins=50, xlim=(0,1250000))
        fig2 = plot_histogramv2(dfNoMissingValues5, 'bed', bins=4, xlim=(0,10))
        fig3 = plot_histogramv2(dfNoMissingValues5, 'bath', bins=4, xlim=(0,5))
        fig4 = plot_histogramv2(dfNoMissingValues5, 'acre_lot', bins=50, xlim=(0,2))
        fig5 = plot_histogramv2(dfNoMissingValues5, 'house_size', bins=75, xlim=(0,4000))
        content = html.Div([
            dcc.Graph(
                id='pricehistogram',
                figure=fig1,
            ),
            dcc.Graph(
                id='bedhistogram',
                figure=fig2
            ),
            dcc.Graph(
                id='bathhistogram',
                figure=fig3
            ),
            dcc.Graph(
                id='acrelothistogram',
                figure=fig4
            ),
            dcc.Graph(
                id='housesizehistogram',
                figure=fig5
            )
        ])
        figures = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5}
        dropdown_options = [{'label':'Price histogram', 'value':'fig1'},
                            {'label':'Bed histogram', 'value':'fig2'},
                            {'label':'Bath histogram', 'value':'fig3'},
                            {'label':'Acrelot histogram', 'value':'fig4'},
                            {'label':'Housesize histogram', 'value':'fig5'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-2':
        fig1 = plot_barchart(dfNoMissingValues5, 'status')
        fig2 = plot_barchart(dfNoMissingValues5, 'state')
        figures = {'fig1':fig1, 'fig2':fig2}
        content = html.Div([
            dcc.Graph(
                id='statusbarchart',
                figure=fig1
            ),
            dcc.Graph(
                id='statebarchart',
                figure=fig2
            )
        ])
        dropdown_options = [{'label':'Status Bar Chart', 'value':'fig1'},
                            {'label':'State Bar Chart', 'value':'fig2'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-3':
        fig = correlation_matrix(dfNoMissingValues5, include_columns)
        content = html.Div([
            dcc.Graph(id='correlation-matrix', figure=fig)
        ])
        figures = {'fig1':fig}
        dropdown_options =[{'label':'Correlation Matrix','value':'fig'}]
        return content, figures, dropdown_options, 'fig'
    elif tab == 'tab-4':
        fig = plot_price_trends(dfcleaneddates, 'prev_sold_date', 'price')
        content = html.Div([
            dcc.Graph(id='pricetrendtime', figure=fig)
        ])
        figures = {'fig1':fig}
        dropdown_options =[{'label':'Time Series Price Trend','value':'fig1'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-5':
        fig = plot_location_price_effects(dfcleaneddates, 'state', 'price')
        content = html.Div([
            dcc.Graph(id='pricetrendbystate', figure=fig)
        ])
        figures = {'fig1': fig}
        dropdown_options = [{'label':'Price Trends by State','value':'fig1'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-6':
        fig1 = plot_size_vs_price(dfcleaneddates, 'house_size', 'price')
        fig2 = plot_size_vs_price(dfcleaneddates, 'acre_lot', 'price')
        figures = {'fig1':fig1, 'fig2': fig2}
        content = html.Div([
            dcc.Graph(
                id='housesizevprice',
                figure=fig1,
                style={'display':'inline-block', 'width': '100%'}
            ),
            dcc.Graph(
                id='acrelotvprice',
                figure=fig2,
                style={'display':'inline-block', 'width':'100%'}
            )
        ])
        dropdown_options = [{'label':'House Size versus Price','value':'fig1'},
                            {'label':'Acre lot versus Price','value':'fig2'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'tab-7':
        fig1 = plot_scatter(dfcleaneddates, 'price', 'bed')
        fig2 = plot_scatter(dfcleaneddates, 'price', 'bath')
        figures = {'fig1':fig1,'fig2':fig2}
        content = html.Div([
            dcc.Graph(
                id='pricevbedscatter',
                figure=fig1,
                style={'display':'inline-block', 'width': '100%'}
            ),
            dcc.Graph(
                id='pricevbathscatter',
                figure=fig2,
                style={'display':'inline-block', 'width':'100%'}
            )
        ])
        dropdown_options = [{'label':'Price v. Bed','value':'fig1'},
                            {'label':'Price v. Bath','value':'fig2'}]
        return content, figures, dropdown_options, 'fig1'
    elif tab == 'linear-regression':
        content = html.Div([
            html.Button("Train and Evaluate Model", id='train-button', n_clicks=0),
            html.Div(id='model-output')
        ])
        return content, {}, [], None
    elif tab == 'random-forest':
        content = html.Div([
            html.Button("Train and Evaluate Model", id='train-rf-button', n_clicks=0),
            html.Div(id='rf-model-output'),
            html.Div(id='rf-model-runtime'),
            dcc.Interval(id='rf-timer-interval', interval=1000, n_intervals=0, max_intervals=0)
        ])
        return content, {}, [], None
    elif tab == 'gradient-boosting':
        content = html.Div([
            html.Button("Train and Evaluate Gradient Boosting", id='train-gbm-button', n_clicks=0),
            html.Div(id='gbm-model-output'),
            html.Div(id='gbm-model-runtime'),
            dcc.Interval(id='gbm-timer-interval', interval=1000, n_intervals=0, max_intervals=0)
        ])
        return content, {}, [], None
    
@app.callback(
    Output("model-output", "children"),
    Input("train-button", "n_clicks"),
    prevent_initial_call=True
)
def update_model_output(n_clicks):
    if n_clicks:
        results = train_evaluate_LinearRegression(X_train, X_test, Y_train, Y_test)
        return html.Div([
            html.P(f"Mean Squared Error: {results['mse']:.2f}"),
            html.P(f"R^2 Score: {results['r2']:.3f}")
        ])
    return "Click the button to train and evaluate the Linear Regression model."

@app.callback(
        [Output('rf-model-runtime', 'children'),
         Output('rf-timer-interval', 'max_intervals')],
         [Input('train-rf-button', 'n_clicks'),
          Input('rf-timer-interval', 'n_intervals')],
          State('rf-timer-interval', 'max_intervals'),
          prevent_initial_call=True
)
def update_rf_timer(n_clicks, n_intervals, max_intervals):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'train-rf-button':
        return "Running time: 0 seconds", 1
    elif triggered_id == 'rf-timer-interval' and max_intervals != -1:
        return f"Running time: {n_intervals} seconds", no_update
    return no_update, -1

@app.callback(
        Output("rf-model-output", "children"),
        [Input("train-rf-button", "n_clicks")],
        prevent_initial_call=True
)
def update_rf_model_output(n_clicks):
    if n_clicks:
        start_time = time.time()
        results = train_evaluate_randomforest(X_train, X_test, Y_train, Y_test)
        elapsed_time = time.time() - start_time
        return html.Div([
            html.P(f"Random Forest MSE: {results['mse']:.2f}"),
            html.P(f"Random Forest R^2 Score: {results['r2']:.3f}"),
            html.P(f"Elapsed Time: {elapsed_time:.2f} seconds")
        ])
    raise PreventUpdate

@app.callback(
        [Output('gbm-model-runtime', 'children'),
         Output('gbm-timer-interval', 'max_intervals')],
         [Input('train-gbm-button', 'n_clicks'),
          Input('gbm-timer-interval', 'n_intervals')],
          State('gbm-timer-interval', 'max_intervals'),
          prevent_initial_call=True
)
def update_gbm_timer(n_clicks, n_intervals, max_intervals):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'train-gbm-button':
        return "Running time: 0 seconds", 1
    elif triggered_id == 'gbm-timer-interval' and max_intervals != -1:
        return f"Running time: {n_intervals} seconds", no_update
    return no_update, -1
    
        
@app.callback(
        Output("gbm-model-output", "children"),
        [Input("train-gbm-button", "n_clicks")],
        prevent_initial_call=True)

def update_gbm_model_output(n_clicks):
    if n_clicks:
        start_time = time.time()
        results = train_evaluate_gbm(X_train, X_test, Y_train, Y_test)
        elapsed_time = time.time() - start_time
        return html.Div([
            html.P(f"Gradient Boosting MSE: {results['mse']:.2f}"),
            html.P(f"Gradient Boosting R^2 Score: {results['r2']:.3f}"),
            html.P(f"Elapsed Time: {elapsed_time:.2f} seconds")
    ])
    raise PreventUpdate

@app.callback(
    Output('download-button-container', 'children'),
    [Input('figure-select-dropdown', 'value')],
    [State('figure-store', 'data')]
)
def update_download_button(selected_figure_id, figures):
    if selected_figure_id and figures:
        fig_data = figures.get(selected_figure_id)
        if fig_data:
            return DownloadButton(app, "Download Selected Graph", f"{selected_figure_id}.png").layout()
        return "Select a graph to download"


if __name__ == '__main__':
    app.run_server(debug=True)
