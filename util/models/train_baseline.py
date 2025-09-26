from sklearn.linear_model import LinearRegression,Lasso,ElasticNet

def train_baseline_MLR(X,y):
    regr = LinearRegression()
    regr.fit(X['train'],y['train'])
    return regr