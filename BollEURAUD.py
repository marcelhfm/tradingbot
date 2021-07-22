from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.types as tp
import pandas as pd
import numpy as np
import datetime as dt
import json
import API_KEYS

class BollingerEURUSD():
    def __init__(self, access_token, accountID, instrument, bar_length, units):
        self.access_token = access_token
        self.accountID = accountID
        self.position = 0
        self.instrument = instrument
        self.bar_length = bar_length
        self.units = units
        self.devs = 2
        self.sma_window = 20
        self.tick_data = pd.DataFrame()
        self.hist_data = pd.DataFrame()
        self.min_length = None
        self.raw_data = None
        self.tp_id = None
        self.sl_id = None
        self.trade_id = None
        self.order_price = None
        self.sl_changed = False


        self.client = API(access_token=self.access_token)

    def get_most_recent(self, days=5):
        '''
        retrieve historical data.

        params:
        days=amount of days to retrieve. Default=5
        '''

        #Create start and end point for retrieving data
        now = dt.datetime.utcnow()
        now = now - dt.timedelta(microseconds= now.microsecond)
        now = now - dt.timedelta(hours=5)
        past = now - dt.timedelta(days=days)
        from_date = past.strftime('%Y-%m-%dT%H:%M:%S')
        to_date = now.strftime('%Y-%m-%dT%H:%M:%S')

        #params
        params={"granularity":"M5","from": from_date, "to": to_date}

        #retrieve data
        r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
        rv = self.client.request(r)

        #save data in dataframe
        for _ in rv["candles"]:
            df = pd.DataFrame({self.instrument: float(_["mid"]["c"])}, index = [pd.to_datetime(_["time"])])
            self.hist_data = self.hist_data.append(df)

        #resample to desired bar_length
        self.hist_data = self.hist_data.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
        self.min_length = len(self.hist_data) + 1

    def resample_and_join(self):
        '''
        resample and append data to existing dataframe
        '''
        self.raw_data = self.hist_data.append(self.tick_data.resample(self.bar_length, label="right").last().ffill().iloc[:-1])

    def prepare_data(self):
        '''
        Every new bar calculate helper columns for the algorithm
        '''
        df = self.raw_data.copy()
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        df["sma"] = df[self.instrument].rolling(self.sma_window).mean()
        df["upper"] = df.sma + df[self.instrument].rolling(self.sma_window).std() * self.devs
        df["lower"] = df.sma - df[self.instrument].rolling(self.sma_window).std() * self.devs
        df.dropna(inplace=True)
        return df

    def check_position(self):
        '''
        check if any trades are currently open, change self.position accordingly

        if trade is open, print unrealized profit
        '''
        s = trades.OpenTrades(accountID=self.accountID)
        sv = self.client.request(s)

        if len(sv["trades"]) == 0:
            #no open trades
            self.position = 0
            self.sl_changed = False
        else:
            #set position accordingly
            if int(sv["trades"][0]["initialUnits"]) > 0 and sv["trades"][0]["state"] == 'OPEN':
                self.position = 1

                #print unrealized profit
                a = accounts.AccountDetails(accountID=self.accountID)
                av = self.client.request(a)
                unrealized_pl = av["account"]["positions"][1]["long"]["unrealizedPL"]
                print("\nUnrealized Profit in LONG position: {}".format(unrealized_pl))
            elif int(sv["trades"][0]["initialUnits"]) < 0 and sv["trades"][0]["state"] == 'OPEN':
                self.position = -1

                #print unrealized profit
                a = accounts.AccountDetails(accountID=self.accountID)
                av = self.client.request(a)
                unrealized_pl = av["account"]["positions"][1]["short"]["unrealizedPL"]
                print("\nUnrealized Profit in SHORT position: {}".format(unrealized_pl))

            #change sl if unrealized_pl > 20
            if float(unrealized_pl) > 20:
                self.change_sl(self.order_price)

    
    def change_sl(self, price):
        '''
        Change stop loss in existing trade to "price"
        '''

        if self.sl_changed == False:

            data = {
                "order": {
                   "type": "STOP_LOSS",
                   "tradeID": self.trade_id,
                    "price": tp.PriceValue(price).value,
                   "timeInForce": "GTC",
                    "triggerCondition": "DEFAULT"
               }
            }

            sl = orders.OrderReplace(accountID=self.accountID, data=data, orderID=self.sl_id)
            sv = self.client.request(sl)

            self.report_trade(price=price, going_direct="CHANGED STOP LOSS", time= sv["orderCreateTransaction"]["time"], units=0)

            self.sl_changed = True

    def create_order(self, going, multi=1):
        '''
        Create and place order

        params:
        going=String object, either "SHORT", "LONG" or "NEUTRAL"
        multi = int, either 1 or 2, default=1, if going == "NEUTRAL" multi is either +1 or -1

        '''
        if going == "SHORT" or going == "LONG":
            if going == "SHORT":
                if multi == 1:
                    coef = -1
                else:
                    coef = -2
                tp_price = tp.PriceValue(self.bid - 2.2 * 0.001).value
            
            elif going == "LONG":
                if multi == 1:
                    coef = 1
                else:
                    coef = 2
                tp_price = tp.PriceValue(self.ask + 2.2 * 0.001).value
            
            #create data dict
            data = {
                "order": {
                    "type": "MARKET",
                    "instrument": self.instrument,
                    "units": coef * self.units,
                    "timeInForce": "FOK",
                    "positionFill": 'DEFAULT',
                    "takeProfitOnFill": {
                        "price": tp_price,
                        "timeInForce": "GTC"
                    },
                    "stopLossOnFill": {
                        "distance": tp.PriceValue(0.0013).value,
                        "timeInForce": "GTC"
                    }
                }
            }
        #if going neutral:
        else:
            #creata data dict
            data = {
                "order": {
                    "type": "MARKET",
                    "instrument": self.instrument,
                    "units": multi * self.units,
                    "timeInForce": "FOK",
                    "positionFill": 'DEFAULT'
                }
            }
        #execute transaction
        o = orders.OrderCreate(accountID=self.accountID, data=data)
        ov = self.client.request(o)
        self.sl_id = ov["relatedTransactionIDs"][-1]
        self.tp_id = ov["relatedTransactionIDs"][-2]
        self.trade_id = ov["relatedTransactionIDs"][-3]
        self.order_price = ov["orderFillTransaction"]["price"]
        #report trade
        self.report_trade(price = self.order_price,
         going_direct="GOING " + going, 
          time = o.response["orderFillTransaction"]["time"],
           units = o.response["orderFillTransaction"]["units"])

        


    def start_stream(self):
        '''
        Start streaming data, aka start trading algorithm
        '''
        params = {"instruments":self.instrument}

        r = pricing.PricingStream(accountID=self.accountID, params=params)
        rv = self.client.request(r)

        for tick in rv:
            #renew dataframe with recent data
            if tick["type"] == 'PRICE':
                self.ask = float(tick["closeoutAsk"])
                self.bid = float(tick["closeoutBid"])
                df = pd.DataFrame({self.instrument: (self.ask + self.bid) / 2}, index = [pd.to_datetime(tick["time"])])
                self.tick_data = self.tick_data.append(df)
                self.resample_and_join()
                
                #Only if new bar has been added
                if len(self.raw_data) > self.min_length - 1:
                    self.min_length += 1

                    #check position and printout unrealized PL
                    self.check_position()

                    #prepare data
                    self.data = self.prepare_data()

                    #printout for error checking
                    print("\n" + "Price: {} | Upper: {} | Lower: {} | SMA: {} \n".format(self.data[self.instrument].iloc[-1],
                     self.data.upper.iloc[-1] < self.data[self.instrument].iloc[-1],
                      self.data.lower.iloc[-1] > self.data[self.instrument].iloc[-1],
                       self.data.sma.iloc[-1] < self.data[self.instrument].iloc[-1]))
                    
                    #Trading algorithm
                    #neutral position
                    if self.position == 0:
                        #second last bar is above upper and last bar is declining
                        if self.data[self.instrument].iloc[-2] > self.data["upper"].iloc[-2] and np.sign(self.data.returns.iloc[-1]) < 0:
                        #price has yet not crossed sma again
                            if not self.data[self.instrument].iloc[-1] < self.data.sma.iloc[-1]:
                                #Create order: GOING SHORT
                                self.create_order("SHORT", multi=1)
                                self.position = -1
                        #second last bar is below lower and last bar is climbing
                        elif self.data[self.instrument].iloc[-2] < self.data["lower"].iloc[-2] and np.sign(self.data.returns.iloc[-1]) > 0:
                            #price has yet not crossed sma again
                            if not self.data[self.instrument].iloc[-1] > self.data.sma.iloc[-1]:
                                #create Order: GOING LONG  
                                self.create_order("LONG", multi=1)
                                self.position = 1

                    #short position 
                    elif self.position == -1:
                        #price has crossed sma
                        if self.data[self.instrument].iloc[-1] < self.data["sma"].iloc[-1]:
                            #price has crossed lower
                            if self.data[self.instrument].iloc[-1] < self.data["lower"].iloc[-1]:
                                #Create order: GOING LONG      
                                self.create_order("LONG", multi=2)
                                self.position = 1
                            else:
                                #Create order: GOING NEUTRAL
                                self.create_order("NEUTRAL", multi=1)
                                self.position = 0
                    
                    #long position
                    elif self.position == 1:
                        #price has crossed sma
                        if self.data[self.instrument].iloc[-1] > self.data["sma"].iloc[-1]:
                            #price has crossed upper
                            if self.data[self.instrument].iloc[-1] > self.data["upper"].iloc[-1]:
                                #Create order: GOING SHORT
                                self.create_order("SHORT", multi=2)
                                self.position = -1
                            else:
                                #Create order: GOING NEUTRAL
                                self.create_order("NEUTRAL", multi=-1)
                                self.position = 0

    def report_trade(self, price, going_direct, time, units):
        '''
        printout to console after order has been created

        params:
        price = price order has been created at

        going = Long/Short

        time = time order has been created at

        units = amount of units used in order
        '''
        print("\n" + 100 * "-")
        print("{} | {}".format(time, going_direct))
        print("{} | units = {} | price = {} |".format(time, units, price))
        print(100 * "-" + "\n")



def main():
    trader = BollingerEURUSD(API_KEYS.API_KEY, accountID = API_KEYS.accountID_5, instrument="EUR_AUD", bar_length="15min", units=30000)


    trader.get_most_recent()
    trader.start_stream()

if __name__ == "__main__":
    main()



