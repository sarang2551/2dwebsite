from app.serverlogic import LinearRegression
import pandas as pd
import numpy as np
import plotly.express as plt

def train_and_predict(model:LinearRegression,vector_dict)->None:
    df = pd.read_excel(r"app\regression_data\main_df_raw.xlsx")
    df = df.iloc[:,1:]
    features = []
    means = {}
    stds = {}
    for c in df.columns:
        if c != "Country/Territory":
            if c != "imports":
                features.append(c)
            # populate means and std
            means[c] = df[c].mean()
            stds[c] = df[c].std()
            df[c] = normalize_z(df[c])
    features = df[features]
    target = df["imports"]
    x_train,x_test,y_train,y_test = split_data(features,target,test_size=0.2,random_state=661)
    features = prepare_feature(x_train)
    target = prepare_target(y_train)
    model.fit(features,target)
    vector = normalize_vector_dict(vector_dict, means, stds)
    normalised_import = (float(vector_dict["current_imports"])-means["imports"])/stds["imports"]
    make_graph(normalised_import,df,vector)
    y_pred = model.predict(process_vector(vector))[0]
    return y_pred

def make_graph(y_norm,df,vector_dict):
    # df is normalised
    # current imports against every other feature
    y_array = df["imports"]
    for col in df.columns:
        if col != "Country/Territory" and col != "imports":
            x_array = df[col]
            x_point_norm = vector_dict[col]
            graph = plt.scatter(df,y="imports",x=col)
            graph.add_scatter(x=[x_point_norm],y=[y_norm],name="You are here")
            graph.write_html(f"./app/templates/{col}.html")


def normalize_vector_dict(vector_dict,mean_dict,std_dict):
    for col in mean_dict.keys():
        if col != "imports":
            m = mean_dict[col]
            s = std_dict[col]
            vector_dict[col] = (float(vector_dict[col])-m)/s
    return vector_dict

def process_vector(data):
    population = data['mean_population']
    arable_land = data['mean_arable']
    cri_score = data['cri_score']
    agriculture_credit = data['credit_agriculture_millions']
    local_agriculture = data['local_agriculture']
    political_index = data['mean_political']

    return np.array([
        agriculture_credit,
        cri_score,
        local_agriculture,
        population,
        political_index,
        arable_land
        ])
# normalize all the columns in a dataframe
def normalize_z(df):
    df = pd.DataFrame(df)
    if isinstance(df,pd.DataFrame):
        for col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            df[col] = (df[col]-m)/s
        return df


def prepare_feature(df_feature):
    if len(df_feature.shape) == 2:
        r,c = df_feature.shape
    else:
        r = len(df_feature)
        c = 1
    df_feature = df_feature.to_numpy().reshape(r,c)
    val = np.concatenate((np.ones(shape=(r,1)),df_feature),axis=1)
    return val

def prepare_target(df_target):
    if len(df_target) == 2:
        r,c = df_target.shape
    else:
        r = len(df_target)
        c = 1
    if c > r:
        return df_target.to_numpy().reshape(c,r)
    return df_target.to_numpy().reshape(r,c)
    

def split_data(df_feature, df_target, random_state=100, test_size=0.3):
    idx = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    k = int(test_size * len(idx))
    test_index = np.random.choice(idx, k , replace = False)
    idx = set(idx)
    test_index = set(test_index)
    train_index = idx - test_index
    df_feature_train = df_feature.loc[list(train_index)]
    df_target_train = df_target.loc[list(train_index)]
    df_feature_test = df_feature.loc[list(test_index)]
    df_target_test = df_target.loc[list(test_index)]
    return df_feature_train, df_feature_test, df_target_train,  df_target_test
