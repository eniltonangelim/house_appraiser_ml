from numpy import absolute
from numpy import arange
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

URL = './dataset/housing.csv'
new_data = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]


def dataframe():
    return read_csv(URL, header=None)

def summarize(df):
    print(df.head())
    print('Shape:', df.shape)


def evaluate_mae(model, X, y):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluation model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


def grid_search_hyperparameters(model, X, y):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, y)
    # Summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)


def predict(model, X, y):
    # fit model
    model.fit(X, y)
    # New Data
    row = new_data
    # Make a prediction
    yhat = model.predict([row])
    # summarize prediction
    print('Predicted: %.3f' % yhat)


def ridge_regression_algo(X, y):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)
    predict(model, X, y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = dataframe()
    summarize(df)

    data = df.values
    X, y = data[:, :-1], data[:, -1]
    # define model
    model = Ridge()
    # Evaluation method
    evaluate_mae(model, X, y)
    # Summarize prediction
    predict(model, X, y)
    # Grid search hyperparameters for ridge regression
    grid_search_hyperparameters(model, X, y)
    # use automatically configured the ridge regression algorithm
    ridge_regression_algo(X, y)