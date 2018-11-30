# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
from sklearn.externals import joblib
from auxiliary import Deal_record, check_balance_warning, get_current_avg_price, get_history_avg_price, generate_bar
import pandas as pd
import numpy as np
import random
from keras.models import load_model
import pickle

# System Settings
ASSETS = ['BCH-USD', 'BTC-USD', 'ETH-USD', 'LTC-USD']
ORIGIN_FEATURES = ['close', 'high', 'low', 'open', 'volume']
asset_names = ['BCH', 'BTC', 'ETH', 'LTC']
LONG, SHORT, HOLD = [1, 0, -1]

# User Settings
CONFIDENCE_SCOPE = 0.005
EXPECT_RETURN_RATE = 0.05
DECISION_PERIOD = 60      # Number of minutes to generate next new bar
BAR_LENGTH = 120

CASH_BALANCE_LOWER_LIMIT = 20000    # cash balance problem!! if my cash balance rise back to 10,000 after sell the crypto, I still cannot make any deal.
CRYPTO_BALACE_LIMIT = 80000

PRIOR = 0.7     # the assumed chance that price will go up/down next week. get by analysis historcal data
PRIOR_WEIGHT = 0.2      # piror weight for each deal. 

# assumeed fluctunate volumn for each DECISION_PERIOD. get by analysis historical data. and should fit for each different currency
FLUCTUATE_VOLUMNS = [15, 100, 6, 1.3]
DEAL_UNITS = [6, 1, 8, 12]      # unit amount for each deal DECISION_PERIOD origin: [3, 1, 5, 10]

# Models Settings

# loal model here, and append models into model List
model_list = []
# for model_name in ASSETS:
#     # model_path = f'./model_{model_name.lower()}.joblib'
#     # model_list.append(joblib.load(model_path))
#     with open('model_'+model_name.split('-')[0]+'.pickle', 'rb') as f:
#         model_list.append(pickle.load(f))
with open('model_BCH.pickle', 'rb') as f:
    model_BCH = pickle.load(f)
    model_list.append(model_BCH)
with open('model_BTC.pickle', 'rb') as f:
    model_BTC = pickle.load(f)
    model_list.append(model_BTC)
with open('model_ETH.pickle', 'rb') as f:
    model_ETH = pickle.load(f)
    model_list.append(model_ETH)
with open('model_LTC.pickle', 'rb') as f:
    model_LTC = pickle.load(f)
    model_list.append(model_LTC)

# decision_count=0
# failed_count = 0
# random.seed(123)
# is_reverse = False

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use

def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # The idea of my strategy:
    # Logistic regression with label = rising/falling signal. 

    # Pattern for long signal:
    # for rising/falling prediction, long/short one unit of & according to the confidence to calculate the goal price, once it reachs sell the crypto.

    # TODO: embeded utility functions
    def get_income_rate():
        pass

    def get_confidence():
        pass

    # memory init
    if counter == 0:
        # data_save only saves data in the latest decision period (1 hour)
        memory.data_save = dict.fromkeys(ASSETS, pd.DataFrame(columns = ORIGIN_FEATURES))
        # memory.old_data = dict.fromkeys(ASSETS)     # bakck up of data_save after one period
        memory.deal_save = dict.fromkeys(ASSETS, [])
        memory.turning_price = dict.fromkeys(ASSETS, 0)     
        memory.volumn_price = dict.fromkeys(ASSETS, 0)      # total money spent on the cryptos

        memory.is_satisfied = False     # If True, stop to make deal
        memory.models_cof = [1 for i in range(len(model_list))]     # confidence for each model
        memory.use_model = [True for i in range(len(model_list))]   # use model or not (this is for ensembled model)

        memory.success_count = dict.fromkeys(ASSETS, 0)

        memory.data_save = []
        memory.data_cryp = []
        for i in list(range(4)):
            #memory.data_save = np.zeros((DECISION_PERIOD, 5))#, dtype=np.float64)
            memory.data_save.append(pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume']))
            memory.data_cryp.append(pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume']))
        memory.hourly = pd.DataFrame(columns = ['rate_BCH', 'dVolume_BCH', 'rate_BTC', 'dVolume_BTC',
                                                'rate_ETH', 'dVolume_ETH', 'rate_LTC', 'dVolume_LTC'])

    
    # data preprocess & record update
    position_new = position_current
    average_prices = np.mean(data[:, :4], axis=1)       # average price for all assets

    if total_balance >= init_cash*(1+EXPECT_RETURN_RATE):
        memory.is_satisfied = True
    else:
        memory.is_satisfied = False

    
    if ((counter + 1) % DECISION_PERIOD == 0):
        for i in list(range(4)):
            memory.data_save[i].loc[DECISION_PERIOD - 1] = data[i, ]
            line = generate_bar(memory.data_save[i])
            memory.data_cryp[i] = pd.concat((memory.data_cryp[i], line), axis=0)
            memory.data_cryp[i] = memory.data_cryp[i].reset_index(drop=True)

    else:
        for i in list(range(4)):
            memory.data_save[i].loc[(counter + 1) % DECISION_PERIOD - 1] = data[i,]

    df_cryp_sp = []
    for i in list(range(4)):
        df_cryp = memory.data_cryp[i].copy()
        if len(df_cryp)>1:

            df_cryp['rate'] = (df_cryp['close'] -df_cryp['open']) / df_cryp['open']
            volume = pd.DataFrame(df_cryp['volume'][0:len(df_cryp)-1], columns=['volume'])
            df_cryp = df_cryp.drop(0).reset_index(drop = True)
            df_cryp['dVolume'] = abs((df_cryp['volume'] - volume['volume']) / volume['volume'])
            df_cryp = df_cryp[['rate', 'dVolume']]
            rateName = 'rate_' + asset_names[i]
            dVolumeName = 'dVolume_' + asset_names[i]
            df_cryp.columns = [rateName, dVolumeName]
            df_cryp_sp.append(df_cryp)
            memory.data_cryp[i] = memory.data_cryp[i].drop(0).reset_index(drop=True)
    if df_cryp_sp:
        cryp_data = pd.DataFrame()
        for cryp in df_cryp_sp:
            cryp_data = pd.concat((cryp_data, cryp), axis=1)
        memory.hourly = pd.concat((memory.hourly, cryp_data), axis=0).reset_index(drop=True)

    y = []

    if len(memory.hourly)>2:
        df_hourly = memory.hourly.copy()
        for name in asset_names:
            bins = [-1, -0.005, 0.01, 1]
            group_name = ['down', 'middle', 'up']
            predName = 'pred_' + name
            rateName = 'rate_' + name
            df_hourly[predName] = pd.cut(df_hourly[rateName], bins, labels=group_name)
            bins_volume = [0, 2.01, 100]
            group_volume_name = ['flat', 'sharp']
            dVolumeName = 'dVolume_' + name
            df_hourly[dVolumeName] = pd.cut(df_hourly[dVolumeName], bins_volume, labels=group_volume_name)
        df_hourly = df_hourly[
            ['pred_BCH', 'pred_BTC', 'pred_ETH', 'pred_LTC', 'dVolume_BCH', 'dVolume_BTC', 'dVolume_ETH',
             'dVolume_LTC']]
        df_cryp_t0 = pd.DataFrame(df_hourly.loc[0:len(memory.hourly) - 3])
        df_cryp_t0.columns = ['BCH_t0', 'BTC_t0', 'ETH_t0', 'LTC_t0', 'dV_BCH_t0', 'dV_BTC_t0', 'dV_ETH_t0',
                              'dV_LTC_t0']
        df_cryp_t0 = df_cryp_t0.reset_index(drop=True)
        df_cryp_t1 = pd.DataFrame(df_hourly.loc[1:len(memory.hourly) - 2])
        df_cryp_t1.columns = ['BCH_t1', 'BTC_t1', 'ETH_t1', 'LTC_t1', 'dV_BCH_t1', 'dV_BTC_t1', 'dV_ETH_t1',
                              'dV_LTC_t1']
        df_cryp_t1 = df_cryp_t1.reset_index(drop=True)
        df_hourly = df_hourly.drop([0,1]).reset_index(drop=True)
        df_hourly = pd.concat((df_hourly, df_cryp_t0), axis=1)
        df_hourly = pd.concat((df_hourly, df_cryp_t1), axis=1)
        X_train = df_hourly.drop(columns=['pred_BCH', 'pred_BTC', 'pred_ETH', 'pred_LTC'])

        X_train = pd.get_dummies(X_train)
        y_BCH = model_BCH.predict(X_train)
        y_BTC = model_BTC.predict(X_train)
        y_ETH = model_ETH.predict(X_train)
        y_LTC = model_LTC.predict(X_train)
        y = [y_BCH, y_BTC, y_ETH, y_LTC]

        memory.hourly = memory.hourly.drop(0).reset_index(drop=True)

    # for each asset do the predict and make deal.
    for asset_index in range(4):

        # when achieving target income, stop making deals
        # TODO: OR when is_satisfied is true & next prediction is opposite result, clean position for more income
        if memory.is_satisfied:
            position_new[asset_index] = 0
            continue

        asset_name = ASSETS[asset_index]
        fluctuate_volumn = FLUCTUATE_VOLUMNS[asset_index]
        deal_unit = DEAL_UNITS[asset_index]
        average_price = average_prices[asset_index]

        # memory.data_save[asset_name].loc[counter] = data[asset_index,]
        
        # predict in DECISION_PERIOD
        if (len(memory.hourly)>2 and (counter+1) % DECISION_PERIOD == 0 and int((counter+1) / DECISION_PERIOD) > 1):
            # Risk Ananlysis
            if check_balance_warning(cash_balance, crypto_balance, total_balance, CASH_BALANCE_LOWER_LIMIT):
                continue

            # Model Evaluation
            if len(memory.deal_save[asset_name]) > 0:
                last_deal = memory.deal_save[asset_name][-1]
            else:
                last_deal = None
            # if decision_count > 150 and (failed_count*1.0) / decision_count > 0.7 and not is_reverse:
            if last_deal and int(memory.data_save[asset_name][0]['open']/memory.data_save[asset_name][-1]['close']) == last_deal.prediction:
                memory.success_count[asset_name] += 1
            else:
                memory.models_cof[asset_index] -= 0.01   # TODO better strategy needed
            if ((counter+1) % (DECISION_PERIOD*24*2) == 0):
                decision_count = int((counter+1) / DECISION_PERIOD)
                memory.use_model[asset_name] = False if memory.success_count[asset_name] / decision_count <= 0.5 else True

            # Do prediction: use  model to predict or not
            if memory.use_model[asset_index]:
                '''
                What should do here:
                    load data for specified asset from memory.data_save[asset_name]
                    transform data format according to diff models
                    call model predict
                    ensemble result
                    give final prediction: 1: increasing, 0: decreasing, 2: hold the line (it's no matter without 2)
                '''
                # data_x = generate_bar(memory.data_save[asset_name].loc[-1-BAR_LENGTH:])
                # price_change = model_list[asset_index].predict(data_x[['avg']])
                # if abs(price_change) > average_price * CONFIDENCE_SCOPE :     # 1 or 0 or other value
                #     prediction = LONG if price_change > 0 else SHORT
                # else:
                #     prediction = HOLD # hold the line
                if y[asset_index] == 'up':
                    prediction = LONG
                elif y[asset_index] == 'down':
                    prediction = SHORT

            else:   # NOT sure is it a better way to replacce model prediction
                # don't use model. judge trend based on the average price of last DECISION_PERIOD & current price.
                hist_avg_price = get_history_avg_price(memory.data_save[asset_name].drop('volume', axis=1), DECISION_PERIOD)
                curr_avg_price = get_current_avg_price(data[asset_index,][:4])
                # prob_pred = 1
                prediction = 1 if hist_avg_price < curr_avg_price else 0

            # TODO: consider transaction fee and calculate income rate to refine deal unit
            if prediction == LONG:
                deal_type = 'long'
                if position_new[asset_index] > 0:
                    position_new[asset_index] += int(deal_unit * memory.models_cof[asset_index])
                    # Assume that new open = last close or avg
                    memory.volume_price[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price       
                elif position_new[asset_index] == 0:
                    position_new[asset_index] += int(deal_unit * memory.models_cof[asset_index])
                    memory.turning_price[asset_name] = average_price
                    memory.volume_price[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price
                else:
                    position_new[asset_index] = 0
                    memory.volume_price[asset_name] = 0
            elif prediction == SHORT:
                deal_type = 'short'
                if position_new[asset_index] <= 0:
                    position_new[asset_index] -= int(deal_unit * memory.models_cof[asset_index])
                    memory.volume_price[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price
                elif position_new[asset_index] == 0:
                    position_new[asset_index] -= int(deal_unit * memory.models_cof[asset_index])
                    memory.turning_price[asset_name] = average_price
                    memory.volume_price[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price       
                else:
                    position_new[asset_index] = 0
                    memory.volume_price[asset_name] = 0
            else: # HOLD
                deal_type = 'none'

            # record deal
            deal_price = memory.data_save[asset_name][-1]['close'] # unknown yet, for now just a assumed close price in this minute
            deal = Deal_record(amount=int(deal_unit * memory.models_cof[asset_index]))
            deal.prob_pred = memory.models_cof[asset_index]
            deal.asset_index = asset_index
            # deal.goal_price = deal_price + (1 if deal_type == 'long' else -1) * (PRIOR_WEIGHT * PRIOR + confidence) * fluctuate_volumn
            deal.deal_type = deal_type
            deal.prediction = prediction
            # deal.has_selled = False
            # deal.has_brought = False
            memory.deal_save.append(deal)
        
        elif(counter != 0):   # not decision period & not the first minute
            # if current currency price can give double expect return, clean position
            # if (position_new[asset_index] > 0) and (average_price >= memory.turning_price[asset_name]*(1+2*EXPECT_RETURN_RATE)):
            #     position_new[asset_index] = 0
            # if (position_new[asset_index] < 0) and (average_price <= memory.turning_price[asset_name]*(1-2*EXPECT_RETURN_RATE)):
            #     position_new[asset_index] = 0
            if (position_new[asset_index] > 0) and (average_price * position_new[asset_index] >= memory.volume_price[asset_name]*(1+1*EXPECT_RETURN_RATE)):
                position_new[asset_index] = 0
            if (position_new[asset_index] < 0) and (average_price * position_new[asset_index] <= memory.volume_price[asset_name]*(1-1*EXPECT_RETURN_RATE)):
                position_new[asset_index] = 0
            
        
    # End of strategy
    return position_new, memory
