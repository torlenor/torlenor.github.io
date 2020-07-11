---
layout: post
title:  "Predicting the outcome of a League of Legends match 10 minutes into the game with 70% accuracy"
date:   2020-07-11 16:00:00 +0200
categories: "Machine Learning"
---

* TOC
{:toc}

Predicting the outcome of a sports match just after a few minutes into the game is an intriguing topic. Wouldn't it be great to know for certain that your favorite football team will win before the game is finished playing? Or betting on your Formula 1 driver to win and being right about it most of the time could earn you a lot of money. While this is not so easily done for regular sports, it can be done for games which heavily depend on the history of the current match, i.e., on things which happened before in the match. Our most promising candidate to develop a model for such a study is Riot Games' League of Legends (LoL).

In this article we will show how to access data from a MongoDB which was fetched from Riot's LoL API, how to process it so that it is usable for modeling, define useful features and how to train an eXtreme Gradient Boosting (XGBoost) model [1]. Using that model we will show that it is possible to predict the outcome of 5v5 Solo Queue match played on Summoner's Rift after just 10 minutes into the match with 70 % accuracy. We will use data from game version 10.13.

The article is written in a hands-on way and we are going to show code examples. As a language of choice we took Python 3 with the libraries matplotlib [2], numpy [3], pandas [4], pymongo [5], seaborn [6], sklearn [7], xgboost [8].

# Introductory remarks

[League of Legends](https://leagueoflegends.com/) (LoL) is a multiplayer online battle arena (MOBA) game. Players compete in matches (in the mode we are looking at 5 vs 5) which can last from 20 to 50 minutes on average (see [Appendix](#appendix) for more details on that). Teams have to work together to achieve victory by destroying the core building (called the Nexus) of the enemy team. Until they get there, they have to destroy towers and get past the defense lines of the enemy team without falling victim of losing their own Nexus in the process.

The players control characters called champions which are picked at the beginning of a match from a rich pool of different champions with their own set of unique abilities. During the mtach the champions will level up and gain additional abilities. They also have to accumulate gold to buy equipment. If a champion is killed it will not permanently die, but just removed from the battle field for a certain amount of time (which grows longer the longer the match is running).

To fetch the data we are using [alolstats](https://github.com/torlenor/alolstats) which provides functionality to fetch match data from [Riot's API](https://developer.riotgames.com/apis) and to store it in a MongoDB collection. It would also feature basic statistical calculations and provide a convenient REST API, but for this project only the ability to fetch and store match data is from importance.

The match data, besides other information, contains timeline information in 0-10 min, 10-20 min, 20-30 min and 30-end min slots for each participant (10 in total, 5 for each team) and we are going to use this data in the modeling approach as features. The prediction target is going to be if team 1 wins the game or now.

As a regression model we are going to use the XGBoost model [1] on approx. 50,000 matches.

# Retrieving data from MongoDB and data preprocessing

Fetching data from a MongoDB is really simple with Python. With just a few lines of code you are receiving a cursor pointing to the data which can be used to iterate through the results. We are taking the results and put them directly into Pandas DataFrames, which may not be the best if we would have a very large collection, but it will do for our data set.

Fetching meta information about the matches from the MongoDB collection (we are filtering for the correct *mapid*, *queueid* and *gameversion* here) can be done via:

```python
import pymongo
import pandas as pd

game_version = "10.13.326.4870"

connection = pymongo.MongoClient("mongodb://[redacted]:[redacted]@localhost/alolstats")
db = connection.alolstats

matches_meta = db.matches.aggregate([
    { "$match": {"gameversion": game_version, "mapid": 11, "queueid": 420}},
    { "$unset": ["teams","participants", "participantidentities"] },
])

df_matches_meta = pd.DataFrame(list(matches_meta))
df_matches_meta = df_matches_meta.set_index("gameid")
```

We will perform the same for the timeline data, but this needs a bit more effort as we have to flatten the embedded documents that we are receiving from our MongoDB collection:


```python
def flatten_nested_json_df(df):
    # Thanks to random StackOverflow user for that piece of code
    df = df.reset_index()

    # search for columns to explode/flatten
    s = (df.applymap(type) == list).all()
    list_columns = s[s].index.tolist()

    s = (df.applymap(type) == dict).all()
    dict_columns = s[s].index.tolist()

    while len(list_columns) > 0 or len(dict_columns) > 0:
        new_columns = []

        for col in dict_columns:
            # explode dictionaries horizontally, adding new columns
            horiz_exploded = pd.json_normalize(df[col]).add_prefix(f'{col}.')
            horiz_exploded.index = df.index
            df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
            new_columns.extend(horiz_exploded.columns) # inplace

        for col in list_columns:
            # explode lists vertically, adding new columns
            df = df.drop(columns=[col]).join(df[col].explode().to_frame())
            new_columns.append(col)

        # check if there are still dict o list fields to flatten
        s = (df[new_columns].applymap(type) == list).all()
        list_columns = s[s].index.tolist()

        s = (df[new_columns].applymap(type) == dict).all()
        dict_columns = s[s].index.tolist()
        
    return df

df_matches_participant = []
for i in range(0,10,1):
    print("Fetching general infos for participant " + str(i+1) + " of 10")
    m = db.matches.aggregate([
        { "$match": {"gameversion": game_version, "mapid": 11, "queueid": 420}},
        { "$addFields": { "participants.gameid": "$gameid" } },
        { "$replaceRoot": { "newRoot": {"$arrayElemAt": [ "$participants", i] }  }  },
        { "$sort" : { "gameid" : 1, "participantid": 1 } },
    ], allowDiskUse = True )
    df_matches_participant.append(flatten_nested_json_df(pd.DataFrame(list(m))).set_index("gameid"))
```

We are ending up with data for each participant of the match, which we can further process to filter out only required columns and limit our features to only timeline fields for 0-10 min:

```python
# Join all participants data into columns so that we have one line per game
X_participants = df_matches_participant[0].join(df_matches_participant[1], lsuffix="_p0", rsuffix="_p1")
for p in range(2,10,1):
    X_participants = X_participants.join(df_matches_participant[p], rsuffix="_p"+str(p))

X_participants_timeline_0_10 = X_participants.filter(regex=("teamid|timeline.*0-10.*"))

# Drop all Diffs between the players on the same lane, we do not want them
X_participants_timeline_0_10 = X_participants_timeline_0_10[X_participants_timeline_0_10.columns.drop(list(X_participants_timeline_0_10.filter(regex='diff')))]

y = pd.DataFrame(df_matches_team1[df_matches_team1["teamid"] == 100]["win"])
y.rename(columns={"win": "team1_did_win"}, inplace=True)
Xy = pd.concat([X_participants_timeline_0_10, y], axis=1)
Xy = X_participants_timeline_0_10.join(y)
Xy = Xy[Xy["team1_did_win"].isnull() == False]

# Final data set for prediction variable...
y_final= Xy["team1_did_win"]
# ... and for features, we drop all data sets were we do now know who one
X_final = Xy.drop('team1_did_win', axis=1)
```

The final data sets look like this now:

- **X_final (first 5 lines):**

    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
        .overflow {
                        overflow-x: scroll;
        }
    </style>
    <div class="overflow">
    <table border="1" class="dataframe" markdown="block">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th>teamid_p0</th>
        <th>timeline.creepspermindeltas.0-10_p0</th>
        <th>timeline.xppermindeltas.0-10_p0</th>
        <th>timeline.goldpermindeltas.0-10_p0</th>
        <th>timeline.damagetakenpermindeltas.0-10_p0</th>
        <th>teamid_p1</th>
        <th>timeline.creepspermindeltas.0-10_p1</th>
        <th>timeline.xppermindeltas.0-10_p1</th>
        <th>timeline.goldpermindeltas.0-10_p1</th>
        <th>timeline.damagetakenpermindeltas.0-10_p1</th>
        <th>teamid</th>
        <th>timeline.creepspermindeltas.0-10</th>
        <th>timeline.xppermindeltas.0-10</th>
        <th>timeline.goldpermindeltas.0-10</th>
        <th>timeline.damagetakenpermindeltas.0-10</th>
        <th>teamid_p3</th>
        <th>timeline.creepspermindeltas.0-10_p3</th>
        <th>timeline.xppermindeltas.0-10_p3</th>
        <th>timeline.goldpermindeltas.0-10_p3</th>
        <th>timeline.damagetakenpermindeltas.0-10_p3</th>
        <th>teamid_p4</th>
        <th>timeline.creepspermindeltas.0-10_p4</th>
        <th>timeline.xppermindeltas.0-10_p4</th>
        <th>timeline.goldpermindeltas.0-10_p4</th>
        <th>timeline.damagetakenpermindeltas.0-10_p4</th>
        <th>teamid_p5</th>
        <th>timeline.creepspermindeltas.0-10_p5</th>
        <th>timeline.xppermindeltas.0-10_p5</th>
        <th>timeline.goldpermindeltas.0-10_p5</th>
        <th>timeline.damagetakenpermindeltas.0-10_p5</th>
        <th>teamid_p6</th>
        <th>timeline.creepspermindeltas.0-10_p6</th>
        <th>timeline.xppermindeltas.0-10_p6</th>
        <th>timeline.goldpermindeltas.0-10_p6</th>
        <th>timeline.damagetakenpermindeltas.0-10_p6</th>
        <th>teamid_p7</th>
        <th>timeline.creepspermindeltas.0-10_p7</th>
        <th>timeline.xppermindeltas.0-10_p7</th>
        <th>timeline.goldpermindeltas.0-10_p7</th>
        <th>timeline.damagetakenpermindeltas.0-10_p7</th>
        <th>teamid_p8</th>
        <th>timeline.creepspermindeltas.0-10_p8</th>
        <th>timeline.xppermindeltas.0-10_p8</th>
        <th>timeline.goldpermindeltas.0-10_p8</th>
        <th>timeline.damagetakenpermindeltas.0-10_p8</th>
        <th>teamid_p9</th>
        <th>timeline.creepspermindeltas.0-10_p9</th>
        <th>timeline.xppermindeltas.0-10_p9</th>
        <th>timeline.goldpermindeltas.0-10_p9</th>
        <th>timeline.damagetakenpermindeltas.0-10_p9</th>
        </tr>
        <tr>
        <th>gameid</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>317415113</th>
        <td>100</td>
        <td>1.5</td>
        <td>230.2</td>
        <td>157.1</td>
        <td>318.2</td>
        <td>100</td>
        <td>0.0</td>
        <td>377.4</td>
        <td>324.2</td>
        <td>849.2</td>
        <td>100</td>
        <td>6.1</td>
        <td>266.1</td>
        <td>230.2</td>
        <td>341.0</td>
        <td>100</td>
        <td>6.3</td>
        <td>368.7</td>
        <td>281.9</td>
        <td>663.4</td>
        <td>100</td>
        <td>7.2</td>
        <td>431.5</td>
        <td>258.7</td>
        <td>482.1</td>
        <td>200</td>
        <td>5.4</td>
        <td>398.7</td>
        <td>233.3</td>
        <td>492.3</td>
        <td>200</td>
        <td>7.3</td>
        <td>511.6</td>
        <td>388.1</td>
        <td>450.8</td>
        <td>200</td>
        <td>7.8</td>
        <td>330.8</td>
        <td>443.3</td>
        <td>351.9</td>
        <td>200</td>
        <td>0.1</td>
        <td>335.3</td>
        <td>342.0</td>
        <td>776.1</td>
        <td>200</td>
        <td>1.3</td>
        <td>328.2</td>
        <td>235.2</td>
        <td>205.3</td>
        </tr>
        <tr>
        <th>317416566</th>
        <td>100</td>
        <td>7.6</td>
        <td>338.3</td>
        <td>263.9</td>
        <td>205.6</td>
        <td>100</td>
        <td>0.3</td>
        <td>274.1</td>
        <td>155.2</td>
        <td>171.7</td>
        <td>100</td>
        <td>4.6</td>
        <td>354.8</td>
        <td>200.6</td>
        <td>419.5</td>
        <td>100</td>
        <td>0.2</td>
        <td>260.6</td>
        <td>230.4</td>
        <td>441.6</td>
        <td>100</td>
        <td>8.3</td>
        <td>499.8</td>
        <td>394.3</td>
        <td>385.4</td>
        <td>200</td>
        <td>7.0</td>
        <td>299.2</td>
        <td>240.6</td>
        <td>157.8</td>
        <td>200</td>
        <td>0.1</td>
        <td>278.3</td>
        <td>132.2</td>
        <td>153.1</td>
        <td>200</td>
        <td>3.7</td>
        <td>371.2</td>
        <td>201.5</td>
        <td>502.1</td>
        <td>200</td>
        <td>8.3</td>
        <td>517.9</td>
        <td>335.8</td>
        <td>211.8</td>
        <td>200</td>
        <td>0.5</td>
        <td>298.1</td>
        <td>258.8</td>
        <td>694.8</td>
        </tr>
        <tr>
        <th>317418523</th>
        <td>100</td>
        <td>5.9</td>
        <td>314.5</td>
        <td>215.4</td>
        <td>397.8</td>
        <td>100</td>
        <td>7.4</td>
        <td>481.4</td>
        <td>259.2</td>
        <td>146.6</td>
        <td>100</td>
        <td>0.0</td>
        <td>310.0</td>
        <td>307.0</td>
        <td>758.9</td>
        <td>100</td>
        <td>0.7</td>
        <td>218.8</td>
        <td>153.1</td>
        <td>241.2</td>
        <td>100</td>
        <td>5.0</td>
        <td>398.7</td>
        <td>196.5</td>
        <td>434.8</td>
        <td>200</td>
        <td>7.3</td>
        <td>489.0</td>
        <td>304.2</td>
        <td>399.2</td>
        <td>200</td>
        <td>1.3</td>
        <td>263.4</td>
        <td>201.9</td>
        <td>269.0</td>
        <td>200</td>
        <td>6.8</td>
        <td>425.8</td>
        <td>301.1</td>
        <td>287.3</td>
        <td>200</td>
        <td>0.6</td>
        <td>364.5</td>
        <td>365.7</td>
        <td>576.5</td>
        <td>200</td>
        <td>9.0</td>
        <td>352.6</td>
        <td>353.6</td>
        <td>259.6</td>
        </tr>
        <tr>
        <th>317419849</th>
        <td>100</td>
        <td>7.3</td>
        <td>423.2</td>
        <td>249.0</td>
        <td>395.0</td>
        <td>100</td>
        <td>0.0</td>
        <td>276.4</td>
        <td>144.3</td>
        <td>24.8</td>
        <td>100</td>
        <td>6.9</td>
        <td>298.6</td>
        <td>270.8</td>
        <td>268.5</td>
        <td>100</td>
        <td>0.4</td>
        <td>311.2</td>
        <td>358.8</td>
        <td>864.9</td>
        <td>100</td>
        <td>5.9</td>
        <td>461.4</td>
        <td>400.6</td>
        <td>519.4</td>
        <td>200</td>
        <td>1.2</td>
        <td>241.1</td>
        <td>167.3</td>
        <td>219.3</td>
        <td>200</td>
        <td>7.5</td>
        <td>344.2</td>
        <td>292.1</td>
        <td>252.3</td>
        <td>200</td>
        <td>3.9</td>
        <td>348.6</td>
        <td>237.9</td>
        <td>635.3</td>
        <td>200</td>
        <td>0.2</td>
        <td>373.8</td>
        <td>366.0</td>
        <td>616.8</td>
        <td>200</td>
        <td>6.1</td>
        <td>435.1</td>
        <td>315.6</td>
        <td>148.0</td>
        </tr>
        <tr>
        <th>317425382</th>
        <td>100</td>
        <td>0.2</td>
        <td>183.0</td>
        <td>177.8</td>
        <td>250.7</td>
        <td>100</td>
        <td>7.5</td>
        <td>499.0</td>
        <td>304.2</td>
        <td>436.4</td>
        <td>100</td>
        <td>0.2</td>
        <td>344.9</td>
        <td>348.8</td>
        <td>671.2</td>
        <td>100</td>
        <td>7.0</td>
        <td>444.3</td>
        <td>235.5</td>
        <td>310.3</td>
        <td>100</td>
        <td>4.0</td>
        <td>341.9</td>
        <td>385.6</td>
        <td>438.5</td>
        <td>200</td>
        <td>0.4</td>
        <td>362.0</td>
        <td>269.1</td>
        <td>676.1</td>
        <td>200</td>
        <td>0.8</td>
        <td>271.6</td>
        <td>286.5</td>
        <td>334.9</td>
        <td>200</td>
        <td>5.7</td>
        <td>273.7</td>
        <td>290.1</td>
        <td>550.2</td>
        <td>200</td>
        <td>3.8</td>
        <td>365.2</td>
        <td>177.4</td>
        <td>485.9</td>
        <td>200</td>
        <td>8.4</td>
        <td>539.3</td>
        <td>294.1</td>
        <td>284.2</td>
        </tr>
    </tbody>
    </table>
    </div>

- **y_final (first 5 lines):**
```
    gameid
    317415113    Fail
    317416566    Fail
    317418523    Fail
    317419849    Fail
    317425382    Fail
    Name: team1_did_win, dtype: object
```

We want to split our data set into training, validation and test sets, to validate the model and to later on test the model on test data. This can be easily accomplished with sklearn train_test_split function:

```python
from sklearn.model_selection import train_test_split
X_tmp, X_test, y_tmp, y_test = train_test_split(X_final, y_final, train_size=0.8, test_size=0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, train_size=0.8, test_size=0.2, random_state = 0)
del X_tmp, y_tmp
```

No data set is perfect and there  are NaN values in the data sets and we have to fill them (the other possibility would be to drop the columns entirely, but we would lose a lot of data). It turns out that Riot seems to set certain fields to NaN if they could not determine certain metrics for a player in that time frame. It is clear that it will not be normally distributed data and we should not use the mean to fill the missing data points. It would be a possibility to take the median to fill the data, but even better works to just set the value to zero. We will use the sklearn SimpleImputer to perform this step:

```python
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
imputed_X_test.columns = X_test.columns

# Imputation removed indices; put them back
imputed_X_train.index = X_train.index
imputed_X_valid.index = X_valid.index
imputed_X_test.index = X_test.index
```

The last step left to do is to encode our prediction target, which is "Win" or "Fail", to something numeric which can be used in our model. We perform this encoding with the LabelEncoder:

```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_y_train = label_encoder.fit_transform(y_train)
label_y_valid = label_encoder.transform(y_valid)
```

# Model definition and fitting

The features we are going to use for the model are now

| Feature | Description |
| --- | --- |
| Team id | The Team ID of that participant (either 100 or 200). |
| Creeps per minute 0-10min | The NPC creatures killed per minute during the time of 0 to 10 minutes into the game. |
| Gold per minute 0-10min | The gold earned per minute during the time of 0 to 10 minutes into the game. |
| Damage taken per minute 0-10min | Damage taken per minute during the time of 0 to 10 minutes into the game. |

These features will occur 10 times in our data set, once for each player in the match.

We define two functions, one for the model and one for judging the quality of the model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def fit_xgboost_model(X_train, y_train, X_valid, y_valid, learning_rate=0.01, n_estimators=500, early_stopping_rounds=5):
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=8)
    model.fit(X_train, y_train,
              early_stopping_rounds=early_stopping_rounds, 
              eval_set=[(X_valid, y_valid)], 
              verbose=False
             )
    return model

def score_dataset(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

To find the best XGB parameters we are looping over number of estimators (n_estimators) and learning rate (learning_rate) and find the parameters which minimize the mean absolute error.

```python
import numpy as np

best_learning_rate = 0
best_n_estimators = 0
best_mae = 100000
for learning_rate in np.arange(0.004, 0.05, 0.002):
    for n_estimators in np.arange(400, 1600, 200):
        model = fit_xgboost_model(imputed_X_train, label_y_train, imputed_X_valid, label_y_valid, learning_rate, n_estimators)
        mae = score_dataset(model, imputed_X_valid, label_y_valid)
        if mae < best_mae:
            best_learning_rate = learning_rate
            best_n_estimators = n_estimators
            best_mae = mae
```

The best parameters for the data set used in this study were `learning_rate = 0` and `n_estimators = 0` and we kept the early stopping rounds at a value of 5.

Using these parameters we perform one final fit of the model which we are going to use for prediction on the test set.

```python
model = fit_xgboost_model(imputed_X_train, label_y_train, imputed_X_valid, label_y_valid, best_learning_rate, best_n_estimators)
```

# Predicting Win/Loss

As we know the real outcome of the matches in the test set, we can compare the predictions

```python
preds_test = model.predict(imputed_X_test)
```

with the actual results


```python
output = pd.DataFrame({"gameid": imputed_X_test.index, "team1_did_win": preds_test})
output["team1_did_win"] = output["team1_did_win"] > 0.5
output = output.set_index("gameid")
```

and calculate the accuracy of our predictions

```python
check = output.join(y_test, lsuffix="_pred", rsuffix="_test")
check["team1_did_win_test"] = (check["team1_did_win_test"] == "Win")

check["equal"] = check["team1_did_win_test"] == check["team1_did_win_pred"]
test_pred_accuracy_percent = len(check[check["equal"] == True]) / len(check) * 100
print("The accuracy on the test set is " + str(test_pred_accuracy_percent) + " %")
```

    The accuracy on the test set is 70.67495737639153 %

As can be seen, we are able to reach an accuracy of 70 % with that simple modeling approach by just taking into account data from the first 10 minutes of the match. This is quite remarkable as it means we are able to correctly predict the outcome of a match after the first 10 minutes more than 2 out of 3 times.

# Summary and discussion

We used the Riot Games API to fetch match data for approx. 50,000 matches for game version 10.13 5v5 Solo Queue on Summoner's Rift. From the fetched match data we extracted features for the first 10 minutes of the match. Using these features we were able to train a model which can predict the winner of the game to an accuracy of 70%, by just looking at the first 10 minutes of that match.

If you have any 

# Appendix

We will take a look at additional metrics which are part of the match data we fetched. We are especially interested in the win rate based on firsts in the match.

Towers are important defensive structures in the game and losing one opens up the possibilities for the other team. As can be seen in the next figure, it can indeed be relevant to lose the first tower.

![](/assets/img/output_33_1.png)
*Figure 1: First Tower Win vs Fail for different regions*

Now it can of course be also interpreted that the result of all what happened before in the match tilted the match into one teams favor making it easier for them to take out a tower. Nevertheless, it can be shown that the win chance and the first tower kill is highly correlated.

In addition, Baron is a very important objective in the game. Not only does it provide a large boost for the team taking it, but it usually also indicates that the match is already going in the favor of that team. In the Win/Fail rate for first Baron kill there is a clear tendency to Win for the team taking the Baron and it is nearly impossible to turn a match if the opposite team takes a Baron.

![](/assets/img/output_34_1.png)
*Figure 2: First Baron Win vs Fail for different regions*

Note: Here the Win/Fail rates do not sum up to 100%, as there are games were neither team takes the Baron. In an extended analysis one should only take matches were the Baron has been taken into consideration for that plot. Nevertheless, it is clear from these numbers, that it is quite hard to turn a game if the opposing team was able to secure the first Baron of the match.

The dragon is an earlier objective and usually not that important in the outcome of a match. Nevertheless, also here it can be seen that getting the first dragon indicates that that team is on a good way to end the match victorious:

![](/assets/img/output_36_1.png)
*Figure 3: First Dragon Win vs Fail for different regions*

The first blood of the match (who killed the a champion of the enemy team first) can happen quite early in a match and from all the investigated metrics this one is the one which does not indicate if a team will win or not that clearly. It seems to be still an indication what team is going to win, however:

![](/assets/img/output_35_1.png)
*Figure 4: First Blood Win vs Fail for different regions*

The last analysis we are going to perform is the average game length. Using the data of all the matches we fetched for game version 10.13 played on Summoner's Rift and in solo 5v5 games we find that, independent of the region, the game duration is approximately close to 30 minutes. We also find the interesting phenomenon that shortly after it is possible to surrender a match, more matches end, which indicates that teams already consider it lost after just 15 minutes into the match. But as we are able to predict the outcome of a game with just 10 minutes of data, those players may indeed be right that they consider the match lost already after just 15 minutes into the match.

![](/assets/img/output_37_1.png)
*Figure 5: Distribution of game length for various regions*

# References

[1] Chen, Tianqi, and Carlos Guestrin. “XGBoost.” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2016), [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754).  
[2] [https://matplotlib.org/](https://matplotlib.org/)  
[3] [https://numpy.org/](https://numpy.org/)  
[4] [https://pandas.pydata.org/](https://pandas.pydata.org/)  
[5] [https://pymongo.readthedocs.io/en/stable/](https://pymongo.readthedocs.io/en/stable/)  
[6] [https://seaborn.pydata.org/](https://seaborn.pydata.org/)  
[7] [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
[8] [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/)  
