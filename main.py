import yfinance as yf 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score




# just looking at time series and prehistoric data
# sp500 = yf.download("^GSPC", period="10y")
sp500 = yf.Ticker("^GSPC").history(period="10y")

# sp500 = yf.Ticker("^GSPC")
# sp500 =  sp500.history(period="10y")

print("SP500")
print(sp500)

print(sp500.index)

sp500.plot.line(y='Close', use_index=True)

# remove irrelevant columns
del sp500["Dividends"]
del sp500["Stock Splits"]


# setup target using ML
sp500["Tomorrow"] = sp500["Close"].shift(-1)

print(sp500)
# print(sp500["Tomorrow"].index)
# print(sp500["Close"].index)
sp500 = sp500.dropna(subset=["Tomorrow"])
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

print(sp500)

# remove data thats from too long ago 
# only take from 1990 onwards
# sp500 = sp500.loc["1990-01-01":].copy()

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# the model to actually learn
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train["Target"])

RandomForestClassifier(min_samples_split=100, random_state=1)


preds = model.predict(test[predictors])


preds = pd.Series(preds, index=test.index)

precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis=1)

combined.plot()


# backtesting system 

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)

counts = predictions["Predictions"].value_counts()
print("counts")
print(counts)

pScore = precision_score(predictions["Target"], predictions["Predictions"])
print("score")
print(pScore)


percentageUp = predictions["Target"].value_counts() / predictions.shape[0]
print("percentage going up")
print(percentageUp)

# additional predictors to improve accuracy
# mean trading close price in days
horizons = [2, 5, 60, 250, 1000]

# holds new cols
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

print("SP500")
print(sp500)
sp500 = sp500.dropna()
print(sp500)

# improving model and prediction to ensure greater accuracy

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    # reduces number of days price goes up increases chance of going up as well
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    

    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)

counts2= predictions["Predictions"].value_counts()
print("count  2")
print(counts2)
# fewer days due to changing the threshold to .6 not .5

pScore2 = precision_score(predictions["Target"], predictions["Predictions"])
print("Pscore 2")
print(pScore2)

