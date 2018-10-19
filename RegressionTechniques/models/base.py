
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import linear_model, metrics
 
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def model_random_forecast(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {
 #       'n_estimators': [500],
 #       'max_features': [10,15],
#	'max_depth': [6,8,10],
 #       'learning_rate': [0.05,0.1,0.15],
  #      'subsample': [0.8]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20]}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


def model_regression_multivariable(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    X_test  = Xtest
    
    y_train = ytrain

    # create linear regression object 
    reg = linear_model.LinearRegression() 

    # train the model using the training sets 
    reg.fit(X_train, y_train)
    
    return reg.predict(X_test), reg.score(X_train, y_train)

def model_regression_multivariable_own(Xtrain, Xtest, ytrain):
    import numpy as np
    X_train = Xtrain
    X_test  = Xtest
    
    y_train = ytrain

    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X_test.mean())/X_test.std()

    theta = np.zeros([1,268])

    def computeCost(X,Y,theta):
        tobesummed = np.power((X_train @ theta.T).sub(y_train, axis=0),2)
        return np.sum(tobesummed)/(2 * len(X))

    def gradientDescent(X, Y, theta,iters,alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            theta = theta - (alpha/len(X)) * np.sum(X * (X_train @ theta.T).sub(y_train, axis=0).T, axis=0)
            cost[i] = computeCost(X, Y, theta)
    
        return theta,cost

    #set hyper parameters
    alpha = 0.01
    iters = 1000

    g,cost = gradientDescent(X_train,y_train,theta,iters,alpha)

    return (X_test @ theta.T), 1