# -*- coding: utf-8 -*-
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

external_stylesheets = [dbc.themes.CERULEAN]
#======================================================================
#           Data retrieval
#======================================================================
filehandler = open("./dat/raw_data","rb")
raw_data=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/raw_dat1","rb")
raw_data1=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/zscore","rb")
zscore=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/iqr","rb")
iqr=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y-test","rb")
y_test=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_test2","rb")
y_test2=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_19","rb")
y_19=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_Gb","rb")
y_pred_GB=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_RF","rb")
y_pred_RF=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_NN","rb")
y_pred_NN=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_XGB","rb")
y_pred_XGB=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_GB19","rb")
y_pred_GB19=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_RF19","rb")
y_pred_RF19=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_NN19","rb")
y_pred_NN19=pickle.load(filehandler)
filehandler.close()

filehandler = open("./dat/y_pred_XGB19","rb")
y_pred_XGB19=pickle.load(filehandler)
filehandler.close()

data_GB=pd.DataFrame(y_test[1:200], columns=['Test'])
data_GB['Prediction'] = y_pred_GB[1:200]
data_XGB=pd.DataFrame(y_test[1:200], columns=['Test'])
data_XGB['Prediction'] = y_pred_XGB[1:200]
data_RF=pd.DataFrame(y_test[1:200], columns=['Test'])
data_RF['Prediction'] = y_pred_RF[1:200]
data_NN=pd.DataFrame(y_test[1:200], columns=['Test'])
data_NN['Prediction'] = y_pred_NN[1:200]

data_GB19=pd.DataFrame(y_19[1:300], columns=['Test'])
data_GB19['Prediction'] = y_pred_GB19[1:300]

data_XGB19=pd.DataFrame(y_19[1:300], columns=['Test'])
data_XGB19['Prediction'] = y_pred_XGB19[1:300]
data_RF19=pd.DataFrame(y_19[1:300], columns=['Test'])
data_RF19['Prediction'] = y_pred_RF19[1:300]
data_NN19=pd.DataFrame(y_19[1:300], columns=['Test'])
data_NN19['Prediction'] = y_pred_NN19[1:300]


#==============Graphs Dynamic====================================================
#EDA
eda1=px.line(raw_data1, x=raw_data1.index, y='Power_kW') 
eda2=px.line(raw_data1, x=raw_data1.index, y='temp_C')
eda3=px.line(raw_data1, x=raw_data1.index, y='solarRad_W/m2')
eda4=px.line(raw_data1, x=raw_data1.index, y='pres_mbar')
eda5=px.line(raw_data1, x=raw_data1.index, y='rain_mm/h')
eda6=px.line(raw_data1, x=raw_data1.index, y='Holiday')

#Clusters


clust1=px.scatter(zscore, x='Power_kW',y='Hour', color='cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"})
clust2=px.scatter(zscore, x='Power_kW',y='temp_C', color='cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"})
clust3=px.scatter(zscore, x='Power_kW',y='solarRad_W/m2', color='cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"})
clust4 = px.scatter_3d(zscore, x='temp_C', y='Hour', z='Power_kW', color='cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"})
clust5 = px.scatter_3d(zscore, x='Power_kW', y='Power-1', z='Hour', color='cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"})

#Regression
#GB
gb1 = px.line(data_GB, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
gb2=px.scatter(data_GB,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
mse_gb =mean_squared_error(y_test,y_pred_GB)
rmse_gb =mean_squared_error(y_test, y_pred_GB, squared=False)
mae_gb = mean_absolute_error(y_test,y_pred_GB)
cvRMSE_gb=rmse_gb/np.mean(y_test)
#RF
rf1 = px.line(data_RF, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
rf2=px.scatter(data_RF,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))

MAE_RF=mean_absolute_error(y_test,y_pred_RF)
MSE_RF=mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)

#XGB
xgb1 = px.line(data_XGB, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
xgb2=px.scatter(data_XGB,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
MAE_XGB=mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)

#NN
nn1 = px.line(data_NN, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
nn2=px.scatter(data_NN,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
MAE_NN=mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)


#GB
gb19 = px.line(data_GB19, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
gb29=px.scatter(data_GB19,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
mse_gb19 =mean_squared_error(y_19,y_pred_GB19)
rmse_gb19 =mean_squared_error(y_19, y_pred_GB19, squared=False)
mae_gb19 = mean_absolute_error(y_19,y_pred_GB19)
cvRMSE_gb19=rmse_gb19/np.mean(y_19)
#RF
rf19 = px.line(data_RF19, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
rf29=px.scatter(data_RF19,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))

MAE_RF19=mean_absolute_error(y_19,y_pred_RF19)
MSE_RF19=mean_squared_error(y_19,y_pred_RF19)  
RMSE_RF19= np.sqrt(mean_squared_error(y_19,y_pred_RF19))
cvRMSE_RF19=RMSE_RF19/np.mean(y_19)

#XGB
xgb19 = px.line(data_XGB19, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
xgb29=px.scatter(data_XGB19,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
MAE_XGB19=mean_absolute_error(y_19,y_pred_XGB19) 
MSE_XGB19=mean_squared_error(y_19,y_pred_XGB19)  
RMSE_XGB19= np.sqrt(mean_squared_error(y_19,y_pred_XGB19))
cvRMSE_XGB19=RMSE_XGB19/np.mean(y_19)

#NN
nn19 = px.line(data_NN19, color_discrete_map = {"Prediction":"blue", "Test":"red"},
                       width=700, height=400, labels=dict(x="Time", y="Load_kW"))
nn29=px.scatter(data_NN19,x ='Test', y = 'Prediction', color_discrete_sequence = ["maroon"],
                         width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
MAE_NN19=mean_absolute_error(y_19,y_pred_NN19) 
MSE_NN19=mean_squared_error(y_19,y_pred_NN19)  
RMSE_NN19= np.sqrt(mean_squared_error(y_19,y_pred_NN19))
cvRMSE_NN19=RMSE_NN/np.mean(y_19)



#======================================================================
# Error Cards
#=======================================================================
cardcvRMS =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("cvRMSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='cvrmse', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })
cardmae =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("MAE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='mae', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           }),
cardmse =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("MSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='mse', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })
cardrmse =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("RMSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='rmse', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })

cardcvRMS1 =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("cvRMSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='cvrmse1', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })
cardmae1 =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("MAE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='mae1', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           }),
cardmse1 =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("MSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='mse1', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })
cardrmse1 =dbc.Card(
   [
    dbc.CardBody(
        [
             # html.H6("Strategies", className="card-subtitle"),
              dbc.ListGroup(
                [html.H5("RMSE", className="strat1"),
                    dbc.ListGroupItem( html.Span(id='rmse1', children=''))
                    ])
              ]
        )
       
                    
     ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "13rem",
          "height":"9rem"
           })





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#F1FFFE",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
     html.Img(src='./assets/IST2_Logo.png',style={'height':'20%', 'width':'100%'}),   
     html.P("South Tower Energy Forecast", className=""),
    #  html.P("Sreejith P Sajeev", className="gsh"),
        html.Hr(),
        html.P(
            "Energy Services Project", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("About", href="/", active="exact"),
                dbc.NavLink("Exploratory Data Analysis", href="/page-EDA", active="exact"),
                dbc.NavLink("Data Cleaning", href="/page-clean", active="exact"),
                dbc.NavLink("Clustering & Features",href="/page-CLT",active="exact"),
                dbc.NavLink("Forecast Models", href="/page-RGR", active="exact"),
                dbc.NavLink("Prediction", href="/page-predict", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
             return [
             html.H3('Energy Forcasting South Tower, Instituto Superior Tecnico'),html.Br(),
             html.H5('Author: Sreejith P Sajeev | ist1101483 | sreejith.sajeev@tecnico.ulisboa.pt'),
             html.Hr(),
             html.P("This dash is executed as a part of Energy Services course at Instituto Superior Tecnico,Lisboa.\
                 The dash primarily represents the steps of devoloping energy forecast models for the South Tower of Instituto Superior Tecnico,Lisbon.\
            The building was assigned randomely. The consumption data from 2017 and 2018 were given. In compliment, met data and holidays data was also provided "),
             html.Br(),
             
            html.H5('Overview of the operations'),html.Hr(),
            html.P("The data was parsed on date and merged. The outliers analysis was done using exploratory data analysis. The observed outliers were removed using \
                2 different methods to compare. The clean data was subjected to clustering to find best features/regressors for the models. 2D and 3D clustering was \
                    executed and data was subjected to a SHAP analysis for feature selection. The selected features were used to model 5 different forecast models. The \
                        models were tested and error coefficients are found and represented. The data from 2019 is represnted in EDA tab. The prediciton models are in \
                            prediciton tab.")
                   
               ]
    elif pathname=="/page-EDA":
        return html.Div([
            html.H3('Exploratory Data Analysis'),
            html.Hr(),
            html.P("Select the option to view the data : 2019"),
            dcc.Dropdown(id='rawdrop',
        options=[
            {'label': 'Power (kW)', 'value': 201},
            {'label': 'Temperature', 'value': 202},
            {'label': 'Solar Radiation', 'value':203},
            {'label': 'Pressure', 'value': 204},
            {'label': 'Rain mm/h', 'value': 205},
            {'label': 'Holidays', 'value': 206},    
              ],
        value=201
        ),
        html.Hr(),
        html.Br(),   
        html.Div(id='raw_dat_g')])
                    
              
            
                  
        
    elif pathname == "/page-clean":
        return [
                html.H3('Data Cleaning Analysis'),
                html.Hr(),
                html.Br(),
            dbc.RadioItems(
        id='radio',
        options=[
            {'label': 'Raw Data Before Cleaning', 'value': 301},
            {'label': 'Data with Removed Outliers (ZScore)', 'value': 302},
            {'label': 'Data with Removed Outliers (IQR)', 'value': 303}
        ], 
        value=301,        
        ),
        html.Div(id='clean_dat_g'),
                ]
    elif pathname == "/page-CLT":
        return [
            html.H3('Clustering'),
            html.Hr(),
            html.H5("2D Clusters"),
            dcc.Dropdown(
                id='clust2d',
                options=[
                    {'label':'Power vs Hour','value':"Hour"},
                    {'label':'Power vs Temperature','value':"Temp"},
                    {'label':'Power vs Solar Radiation','value':"Solar"},
                    
                    ],
                value="Hour"
                ),
            html.Div(id='clustering_2d'),
            html.Hr(),
            html.H5("3D Clusters"),
            dcc.Dropdown(
                id='clust3d',
                options=[
                    {'label':'Power vs Temp vs Hour','value':"temph"},
                    {'label':'Power vs Pow-1 vs Hour','value':"powerh"}
                    ],
                value="temph"
                ),
           html.Div(id='clustering_3d'),
            
            html.H5('Feature Selection'),
            
            html.Div(html.Img(src='./assets/Features.png'))
            
            ]
    elif pathname == "/page-RGR":
        return [
                html.H3('Forecast Models'),
            dcc.Dropdown( 
        id='fordropdown',
        options=[        
            {'label': 'Neural Networks','value': 501},
            {'label': 'Gradient Boosting','value': 502},
            {'label': 'Extreme Gradient Boosting','value': 503},
            {'label': 'Random Forest','value': 504},
            {'label': 'Neural Prophet', 'value':505},
        ], 
        value=501
        ),
        html.Div(id='forecast1'),
        html.Div(id='forecast2'),
        html.Br(),
        html.H5("Error Rates for the Selected model"),
        html.Hr(),
        html.Div([dbc.Row([
                dbc.Col(cardmae,width=3),
                dbc.Col(cardmse,width=3),
                dbc.Col(cardrmse,width=3),
                dbc.Col(cardcvRMS,width=3)
                ])]
            
        )
        
                ]
    elif pathname == "/page-predict":
        return [
                html.H3('Prediction for 2019'),
                html.Hr(),
            dcc.Dropdown( 
        id='predict',
        options=[        
            {'label': 'Neural Networks','value': 601},
            {'label': 'Gradient Boosting','value': 602},
            {'label': 'Extreme Gradient Boosting','value': 603},
            {'label': 'Random Forest','value': 604},
            
        ], 
        value=601
        ),
        html.Div(id='forecast19'),
        html.Div(id='forecast29'),
        html.Br(),
        html.H5("Error Rates for the Selected model"),
        html.Hr(),
        html.Div([dbc.Row([
                dbc.Col(cardmae1,width=3),
                dbc.Col(cardmse1,width=3),
                dbc.Col(cardrmse1,width=3),
                dbc.Col(cardcvRMS1,width=3)
                ])]
            
        )
        
                ]
    # Jumbotron
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
#===================================================================
#App Call Backs
#====================================================================

#pg2
@app.callback(Output('raw_dat_g', 'children'), 
              Input('rawdrop', 'value'))

def render_figure_png(value):
    
    if value == 201:
        return html.Div([dcc.Graph(figure=eda1),
        ])   
    elif value == 202:
         return html.Div([dcc.Graph(figure=eda2),
        ])   
    elif value == 203:
         return html.Div([dcc.Graph(figure=eda3),
        ])   
    elif value == 204:
         return html.Div([dcc.Graph(figure=eda4),
        ])   
    elif value == 205:  
        return html.Div([dcc.Graph(figure=eda5),
        ])       
    elif value == 206:
        return html.Div([dcc.Graph(figure=eda6),
        ])   
 #pg3
@app.callback(Output('clean_dat_g', 'children'), 
              Input('radio', 'value'))          
def render_figure_clean(value):
        
    if value == 301:
        return html.Div([dcc.Graph(
            id='raw-data',
            figure={
                'data':[
                    {'x':raw_data.index,'y':raw_data.Power_kW,'type':'scatter','name':'Raw Data before cleaning'}]})])  
    elif value == 302:
        return html.Div([dcc.Graph(
            id='z-score',
            figure={
                'data':[
                {'x':zscore.index,'y':zscore.Power_kW,'type':'scatter','name':'Cleaned using z score'}]})])     
    elif value == 303:
        return html.Div([dcc.Graph(
            id='z-score',
            figure={
                'data':[
                {'x':iqr.index,'y':iqr.Power_kW,'type':'scatter','name':'Cleaned using iqr'}]})])        
#pg4
@app.callback(Output('clustering_2d', 'children'), 
              Input('clust2d', 'value'))  
 
def  render_figure_cluster(clust):
    if clust=='Hour':
        return(
            html.Div(dcc.Graph(id="clustering1",figure=clust1))
        )
    elif clust=='Temp':
        return(
            html.Div(dcc.Graph(id="clustering2",figure=clust2))
        )
    elif clust=='Solar':
        return(
            html.Div(dcc.Graph(id="clustering3",figure=clust3))
        )    
@app.callback(Output('clustering_3d', 'children'), 
              Input('clust3d', 'value'))  
 
def  render_figure_cluster2(clust):
    if clust=='temph':
        return(
            html.Div(dcc.Graph(id="clustering1",figure=clust4))
        )
    elif clust=='powerh':
        return(
            html.Div(dcc.Graph(id="clustering2",figure=clust5))
        )
#pg5
@app.callback(
  [  Output("forecast1", "children"),
     Output("forecast2","children") ,       
     ],
     [Input("fordropdown", "value")])
 
def  render_figure_reg2(reg):
    if reg==501:
        return[
            html.Div(dcc.Graph(id='nn1',figure=nn1)),
            html.Div(dcc.Graph(id='nn2',figure=nn2))
            
        ]
    elif reg==502:
            return[
            html.Div(dcc.Graph(id='gb1',figure=gb1)),
            html.Div(dcc.Graph(id='gb2',figure=gb2))
            
        ]
    elif reg==503:
        return[
            html.Div(dcc.Graph(id='xgb1',figure=xgb1)),
            html.Div(dcc.Graph(id='xgb',figure=xgb2))
            
        ]
    elif reg==504:
         return[
            html.Div(dcc.Graph(id='rf1',figure=rf1)),
            html.Div(dcc.Graph(id='rf2',figure=rf2))
            
        ]
    elif reg==505:
         return[
            html.Div(html.Img(src='./assets/np1.png',style={'height':'20%', 'width':'65%'})),
            html.Div(html.Img(src='./assets/para.png',style={'height':'20%', 'width':'65%'}))
            
        ]
    
@app.callback(
  [  Output("mae", "children"),
     Output("mse","children") ,
     Output("rmse","children"),
     Output("cvrmse","children"),
         
     ],
     [Input("fordropdown", "value")])
 
def  render_figure_reg1(reg):
    if reg==501:
        return[round(MAE_NN,4),round(MSE_NN,4),round(RMSE_NN,4),round(cvRMSE_NN,4)
        ]
    elif reg==502:
            return[round(mae_gb,4),round(mse_gb,4),round(rmse_gb,4),round(cvRMSE_gb,4)
        ]
    elif reg==503:
        return[round(MAE_XGB,4),round(MSE_XGB,4),round(RMSE_XGB,4),round(cvRMSE_XGB,4)
        ]
    elif reg==504:
        return[round(MAE_RF,4),round(MSE_RF,4),round(RMSE_RF,4),round(cvRMSE_RF,4)
        ]
    elif reg==505:
        return[29.5,'NA','NA','NA']
    
#pg5
@app.callback(
  [  Output("forecast19", "children"),
     Output("forecast29","children") ,       
     ],
     [Input("predict", "value")])
 
def  render_figure_reg2(reg):
    if reg==601:
        return[
            html.Div(dcc.Graph(id='nn19',figure=nn19)),
            html.Div(dcc.Graph(id='nn29',figure=nn29))
            
        ]
    elif reg==602:
            return[
            html.Div(dcc.Graph(id='gb19',figure=gb19)),
            html.Div(dcc.Graph(id='gb29',figure=gb29))
            
        ]
    elif reg==603:
        return[
            html.Div(dcc.Graph(id='xgb19',figure=xgb19)),
            html.Div(dcc.Graph(id='xgb29',figure=xgb29))
            
        ]
    elif reg==604:
         return[
            html.Div(dcc.Graph(id='rf19',figure=rf19)),
            html.Div(dcc.Graph(id='rf29',figure=rf29))
            
        ]
    
    
@app.callback(
  [  Output("mae1", "children"),
     Output("mse1","children") ,
     Output("rmse1","children"),
     Output("cvrmse1","children"),
         
     ],
     [Input("predict", "value")])
 
def  render_figure_reg1(reg):
    if reg==601:
        return[round(MAE_NN19,4),round(MSE_NN19,4),round(RMSE_NN19,4),round(cvRMSE_NN19,4)
        ]
    if reg==602:
            return[round(mae_gb19,4),round(mse_gb19,4),round(rmse_gb19,4),round(cvRMSE_gb19,4)
        ]
    if reg==603:
        return[round(MAE_XGB19,4),round(MSE_XGB19,4),round(RMSE_XGB19,4),round(cvRMSE_XGB19,4)
        ]
    if reg==604:
        return[round(MAE_RF19,4),round(MSE_RF19,4),round(RMSE_RF19,4),round(cvRMSE_RF19,4)
        ]
    
            
   
    
if __name__=='__main__':
    app.run_server(debug=False, port=4000)

