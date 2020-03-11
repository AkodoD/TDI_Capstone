import pandas as pd
import numpy as np
import seaborn as sns
from dateutil.parser import parse
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import pipeline
from sklearn.linear_model import Ridge
import warnings
import os

warnings.filterwarnings('ignore')
sns.set()
plt.rcParams.update({'font.size': 22,'figure.dpi':144})

START_DATE='2001-01-01'
END_DATE = '2020-01-01'
prediction_periods = 14
periods_per_year = 4

class Resample_series(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    This takes in an a data series, takes the log of each entry, converts it to rolling quantiles and
    returns either the rolling quantiles or a periodic sample of the rolling quantiles.
    """
    def __init__(self, resample = False, quantiles = [.5,.75,.9,.99], 
                 frequency = "q", window_size = "548d", cutoff_years = None):
        self.resample = resample
        self.quantiles = quantiles
        self.frequency = frequency
        self.window_size = window_size
        self.cutoff_years = cutoff_years
        
    def fit(self,X,y=None):
        "resamples each quantile at the specified frequency for the specified period"
        return self
    
    def transform(self,df):
        if self.cutoff_years:
            start_date = df.index[-1] - pd.Timedelta(f"{self.cutoff_years}y")
            df = df[start_date:].copy()
            
        df = rolling_resampler(df,quantiles = self.quantiles,apply_log = True, resample_period = self.window_size)
        
        if self.resample == True:
            resampled_rolling_df = pd.DataFrame()
            for column in df:
                resampled_rolling_df[column] = df[column].resample(self.frequency).mean()
            df = resampled_rolling_df.fillna(method='ffill')      
        return df
        
class Prior_hardware(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    initialize with a component database a list of years to return and the quantiles to be returned
    
    the transform method returns the quantiles (pulled from the input component datafarme for the years 
    requested where years are years before the prediction date. 
    """
    def __init__(self,component_df,year_list=[1,2,3]):
        self.df = component_df
        self.year_list = year_list
        
    def fit(self,X,y=None):
        return self

    def transform(self, date_indexes):
        output = []
        for date in date_indexes:
            row = []
            for year in self.year_list:
                window_start = date-pd.Timedelta(days=year*365)
                window_end = window_start + pd.Timedelta(days=365)
                component_dataframe_window = self.df[window_start:window_end]
                for quantile in self.df.columns:
                    row.append(component_dataframe_window[quantile].mean())
            output.append(row)
        return np.array(output)
    
class Linear_indexer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    This class fits a numberline to the dates of a date indexed dataframe. The transform method can then
    be used on any other date indexed dataframe to transform its dates onto the same number line preserving
    the date to value conversion.
    
    """
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        self.train_start_date = X[0]
        self.freq = X.freq
        return self

    def transform(self, X):
        """
        the predict depends on the number of periods from the start of the training set to predict date
        this function must return the number corrisponding to the number of periods between the beginning
        of the training set and the dates on which a prediction is required
        """
        full_date_range = pd.date_range(start = self.train_start_date,
                                        end = X[-1],
                                        freq = self.freq)
        prediction_indices = np.arange(len(full_date_range))[-len(X):]
        return prediction_indices.reshape([-1,1])

class Time_cross_validation:
    def __init__(self,model,data,tested_periods = 10,min_periods = 4):
        """
        data is a pandas dataframe with one column
        tested_periods is how many periods are used to score the fit
        """
        self.model = model
        self.data = data
        self.tested_periods = tested_periods
        self.min_periods = min_periods
        
    def splits(self):
        window_start = self.min_periods
        window_end = len(self.data) - self.tested_periods + 1
        self.train_set = []
        self.test_set = []
        for index in range(window_start, window_end):
            self.train_set.append(self.data[0:index])
            self.test_set.append(self.data[index:index+self.tested_periods])
        
    def score(self):
        mse = []
        self.splits()

        for X,Y in zip(self.train_set,self.test_set):
            self.model.fit(X.index,X.values.reshape([-1,1]))
            predicts = self.model.predict(Y.index)
            ybar = np.mean(Y.values)
            SSres = sum([(fi - yi)**2 for fi,yi in zip(predicts,Y.values.flatten())])
            mse.append(SSres)
        return np.mean(mse) 

def set_time_index(df, date_col):
    """
    df is the dataframe to be re_indexed 
    date_col is a tuple of the the date containing column
    """
    new_df = df.set_index(date_col).copy()
    new_df = new_df.sort_index()
    "add a small time adjustment to prevent any duplicated indices"
    time_steps = pd.to_timedelta(np.arange(len(df)), 'S')
    new_index = new_df.index + time_steps
    new_df.index = new_index
    new_df = new_df[START_DATE:END_DATE]
    return new_df

def rolling_resampler(df, resample_period="548d", quantiles = [.5,.75,.9], apply_log = True):
    """take the log of a data series and finds the rolling avg 
    of the specified period for specified quantiles
    """
    if apply_log:
        df = df.apply(np.log)
    index = [f"r{int(100*quantile)}" for quantile in quantiles]
    return pd.DataFrame([df.rolling(resample_period).quantile(x) for x in quantiles],
                          index = index).transpose()
                          
def  make_graph_lines(input_df, prediction_periods, historical_hardware_years, historical_requirements_years,
                        periods_per_year, fit_pipe):
    "Make the lines for plotting (forecast and history for component type)"
    number_prediction_periods = prediction_periods
    data_daterange = input_df.index

    historical_lines = []
    for quantile in input_df.columns:
        historical_lines.append((data_daterange,input_df[quantile].values.reshape([-1,1])))

    forecast_lines = []
    prediction_daterange = (data_daterange + number_prediction_periods)[-(number_prediction_periods+1):]

    forecast_lines.append((prediction_daterange,[0 for _ in prediction_daterange])) # 0 line for fill

    lookback_buffer_periods = max(historical_hardware_years+historical_requirements_years)*periods_per_year
    fit_daterange = data_daterange[lookback_buffer_periods:]

    for col in input_df.columns:
        fit_Yvals = input_df[col].values[lookback_buffer_periods:].reshape([-1,1])
        predicts = fit_pipe.fit(fit_daterange,fit_Yvals).predict(prediction_daterange).flatten()
        predicts[0] = fit_Yvals[-1] #force the beginning of the precdiction to match the end of the history
        forecast_lines.append((prediction_daterange,predicts))

    forecast_lines.append((prediction_daterange,[20 for _ in prediction_daterange])) # maximum line for fill_between 
    return (historical_lines,forecast_lines,prediction_daterange)

"plot related functions"
def component_labels():
    for x in ['Limited Selection','Most Games','Demanding Games','Nearly Anything']:
        yield(x)
        
def color_gen():
    colors = ['k','xkcd:grey','xkcd:light grey']
    for color in colors:
        yield color
        
def fill_color_gen():
    colors = sns.color_palette("coolwarm_r", 4)
    for color in colors:
        yield color
        
def set_plt_params():
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['xtick.labelsize'] =16
    plt.rcParams['ytick.labelsize'] =16
    plt.rcParams.update({
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"
    })    

def read_preliminary_data(component_type, component_model):
    if component_type == "CPU":
        input_df = pd.read_pickle("./data/preprocessed_data/final_cpu_df.pkl")
        raw_df = input_df
    elif component_type == "GPU":
        input_df = pd.read_pickle("./data/preprocessed_data/final_gpu_df.pkl")
        raw_df = input_df

    req_df = pd.read_excel("./data/preprocessed_data/preprocessed game requirements.xlsx", header = [0,1], index_col = [0])
    return(input_df, raw_df, req_df)

def clean_and_standardize(req_df, input_df, resample, component_type, component_model, raw_df):
    "Convert input dataframes to standard forms"
    "Game Requirements"
    req_df = req_df[[('Game Info',"Release Date"),
                ('Minimum','gpu_tex'),('Recommended','gpu_tex'),
                ('Minimum','flops'),('Recommended','flops')]].copy()

    "set index as a datetime for each dataframe"
    req_df = set_time_index(req_df,('Game Info', 'Release Date'))

    if component_type == "CPU":
        min_cpu_df = req_df['Minimum','flops'].dropna()
        input_df = resample.transform(min_cpu_df)
    elif component_type == "GPU":
        min_gpu_df = req_df['Minimum','gpu_tex'].dropna()
        input_df = resample.transform(min_gpu_df)

    "Hardware components"
    if component_type == "CPU":
        cpu_df = set_time_index(raw_df.reset_index(),"released")
        correlated_df = resample.transform(cpu_df['gflops'])
    elif component_type == "GPU":
        gpu_df = set_time_index(raw_df.reset_index(),"Release Date")
        correlated_df = resample.transform(gpu_df['texture_rate'])
        
    return (input_df, correlated_df)

def optimize_hyperparameters(input_df, correlated_df):
    "parameters for optimization"
    year_lists = [[4],[4,5],[5],[4,5,6],[5,6],[6]]
    alpha_list = [.01,.1,1,10,50]
    
    param_book = {}
    counter = 0
    for historical_hardware_years in year_lists:
        for historical_requirements_years in year_lists:
            for alpha in alpha_list:
                prediction_window_start = (input_df.index +
                               max(historical_hardware_years+historical_requirements_years) * periods_per_year)[0] 

                fu = pipeline.FeatureUnion([('old_hardware',Prior_hardware(correlated_df,year_list = historical_hardware_years)),
                                            ('indexer',Linear_indexer()),
                                            ('old_games',Prior_hardware(input_df,year_list = historical_requirements_years)),
                                           ])

                pipe = pipeline.Pipeline([('fu',fu),
                                                      ('rr',Ridge(alpha = 1))
                                                     ])
                avg_mse = []
                for column in input_df.columns:
                    inp = input_df[column][prediction_window_start:]
                    tt = Time_cross_validation(pipe,inp,min_periods = 10,tested_periods = prediction_periods)
                    avg_mse.append(tt.score())
                param_book[(tuple(historical_hardware_years),
                            tuple(historical_requirements_years),
                            alpha)] = np.mean(avg_mse)
                counter+=1
    #            print(counter/(len(historical_hardware_years)*len(historical_requirements_years)*len(alpha_list)))
    return param_book

def generate_plot_data(input_df, correlated_df, historical_hardware_years, historical_requirements_years, fit_pipe):
    "Make the lines for plotting (forecast and history for component type)"
    data_daterange = input_df.index

    historical_lines = []
    for quantile in input_df.columns:
        historical_lines.append((data_daterange,input_df[quantile].values.reshape([-1,1])))

    forecast_lines = []
    prediction_daterange = (data_daterange + prediction_periods)[-(prediction_periods+1):]

    forecast_lines.append((prediction_daterange,[0 for _ in prediction_daterange])) # 0 line for fill

    lookback_buffer_periods = max(historical_hardware_years+historical_requirements_years)*periods_per_year
    fit_daterange = data_daterange[lookback_buffer_periods:]

    for col in input_df.columns:
        fit_Yvals = input_df[col].values[lookback_buffer_periods:].reshape([-1,1])
        predicts = fit_pipe.fit(fit_daterange,fit_Yvals).predict(prediction_daterange).flatten()
        predicts[0] = fit_Yvals[-1] #force the beginning of the precdiction to match the end of the history
        forecast_lines.append((prediction_daterange,predicts))

    forecast_lines.append((prediction_daterange,[20 for _ in prediction_daterange])) # maximum line for fill_between 
    return (historical_lines, forecast_lines, prediction_daterange)

def create_best_fit_pipeline(input_df, correlated_df, component_type, component_model):
    "best parameters found are hardcoded"
    if component_type == "CPU":
        historical_hardware_years = [4,5]
        historical_requirements_years = [4,5,6]
        alpha = 1
    elif component_type == "GPU":
        historical_hardware_years = [4,5]
        historical_requirements_years = [4,5,6]
        alpha = .5

    prediction_window_start = (input_df.index +
                               max(historical_hardware_years+historical_requirements_years) * periods_per_year)[0] 

    fu = pipeline.FeatureUnion([('old_hardware',Prior_hardware(correlated_df,year_list = historical_hardware_years)),
                                ('indexer',Linear_indexer()),
                                ('old_games',Prior_hardware(input_df,year_list = historical_requirements_years)),
                               ])

    fit_pipe = pipeline.Pipeline([('fu',fu),
                                  ('rr',Ridge(alpha = alpha))
                                 ])
    return (fit_pipe, historical_hardware_years, historical_requirements_years)
    
def make_graph(component_type, component_model):
    "makes the requested graph"

    "input manipulation"
    input_df, raw_df, req_df = read_preliminary_data(component_type, component_model)
    resample = Resample_series(resample = True, quantiles = [.5, .75, .9])
    input_df, correlated_df = clean_and_standardize(req_df, input_df, resample, component_type, component_model, raw_df)
    
    "option to optimize hyperparameters"
    optimize = False
    if optimize == True:
        optimization_results = optimize_hyperparameters(input_df, correlated_df)
    
    "create a model using the best hyperparameters"
    fit_pipe, historical_hardware_years, historical_requirements_years = create_best_fit_pipeline(input_df, correlated_df, component_type, component_model)
    
    "generate the historical and forecasted game requirements"
    historical_lines, forecast_lines, prediction_daterange = generate_plot_data(input_df, correlated_df, historical_hardware_years, historical_requirements_years, fit_pipe)
    
    "Plot Setup"
    set_plt_params()
    color = color_gen()
    labels = component_labels()
    plt.figure(figsize = (16,9))
    try:
        if component_type == 'CPU':
            performance_metric = "GFLOPS"
            performance = np.log(raw_df.loc[raw_df.name==component_model]['gflops'].values[0])
            release_date = raw_df.loc[raw_df.name==component_model].index[0]
        elif component_type == "GPU":
            performance_metric = "Texture Rate (GTexels/s)"
            performance = np.log(raw_df.loc[raw_df.name==component_model]['texture_rate'].values[0])
            release_date = raw_df.loc[raw_df.name==component_model].index[0]
    except IndexError:
        return "Error was raised when looking the requested component up in the component type database. Ensure component is in database and retry."
        
    "plot the figure"
    color = color_gen()
    labels = component_labels()
    for x,y in historical_lines:
        plt.plot(x,y, color = 'xkcd:grey',
                 linewidth = 4)  
        
    color = color_gen()
    for x,y in forecast_lines[1:-1]:
        plt.plot(x,y, color = 'xkcd:grey', linestyle = "--", linewidth = 4)

    color = fill_color_gen()
    y_bottom = forecast_lines[0][1]
    for line in forecast_lines[1:]:
        plt.fill_between(prediction_daterange,y_bottom,line[1], color = next(color), alpha = 0.2)
        y_bottom = line[1]

    hist_x_vals= input_df.index
    y_bottom = [0 for _ in hist_x_vals]
    color = fill_color_gen()

    for line in historical_lines:
        line=line[1].flatten()
        plt.fill_between(hist_x_vals,y_bottom,line, 
                         color = next(color), 
                         label = next(labels),
                         alpha = 0.2)
        y_bottom = line
        
    plt.fill_between(hist_x_vals,y_bottom,[20 for _ in hist_x_vals], 
                     color = next(color),
                     label = next(labels),
                     alpha = 0.2)    

    # vertical line between historical and predictions
    plt.axvline(hist_x_vals[-1], color = 'k', linestyle = '--', linewidth = 4)

    "this block shows the desired processor"
    plt.plot([release_date,prediction_daterange[-1]],
             [performance,performance],
            color = 'xkcd:bright red',
             marker = 'o',
             markerfacecolor = 'k',
             markeredgecolor = 'k',
             markersize = 12,
            linewidth = 4)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="xkcd:very light pink", ec="k", lw=2,alpha = 0.5)
    plt.annotate(component_model, 
                 xy=("2020-05-31", performance+.5),
                 size = 20,
                bbox=bbox_props)

    "historical and predicted text"
    plt.annotate("Historical", 
                 xy=("2018-03-01", .5),
                 size = 22)
    plt.annotate("Predicted", 
                 xy=("2020-05-31", .5),
                 size = 22)
                
    plt.xlim('2010-01-01',prediction_daterange[-1])
    plt.ylim(0,max(performance+1,max([max(predict[1]) for predict in forecast_lines[:-1]])+1))
    plt.legend(prop={'size': 18})
    plt.ylabel(f"Log {component_type} {performance_metric}")
    plt.xlabel("Year of Release")
    plt.title(f"Game Minimum {component_type} Requirements", fontsize = 22, color = "white")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    