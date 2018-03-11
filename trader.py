
# coding: utf-8

# # HW1-AutoTrading

# ## Import package

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split


# ## Functions implement
# - Load .csv files by pandas

# In[ ]:


def load_data(path):
    df = pd.read_csv(path, names=['open', 'high', 'low', 'close'])
    data = df.values
    data[:, 1:] -= data[:, :1]
    return data


# ## Dataset Building
# - Store training data and history data

# In[ ]:


class Datasets:
    def __init__(self, data, interval=5):
        self.data = data
        self.interval = interval
        self.target_regression = None
        self._labeling_regression(data)
        self.target_classification = None
        self._labeling_classification(data)
        
    def _labeling_regression(self, data):
        n = data.shape[0];
        t = np.zeros(n);
        for i in range(n):
            if i < n - 2:
                if i < n - self.interval - 1 :
                    x = self.interval
                else:
                    x = n - i - 2
                t[i] = np.mean(data[i + 1:i + x + 1, 0]) - data[i, 0]
            elif i == n - 2:
                t[i] = data[i + 1, 0] - data[i, 0]
                
        self.target_regression = t

    def _labeling_classification(self, data):
        n = data.shape[0];
        t = np.zeros((n, 2));
        for i in range(n):
            if i < n - 2:
                if i < n - self.interval - 1 :
                    x = self.interval
                else:
                    x = n - i - 2
                t[i, (int) (np.mean(data[i + 1:i + x + 1, 0]) > data[i, 0])] = 1
            elif i == n - 2:
                t[i, (int) (data[i + 1, 0] > data[i, 0])] = 1
            else:
                t[i, 1] = 1
            
        self.target_classification = t
    
    def add_data(self, data):
        tmp = np.zeros((self.data.shape[0] + 1, 4))
        tmp[:self.data.shape[0]] = self.data
        tmp[self.data.shape[0]] = data
        self.data = tmp
        self._labeling_regression(self.data)
    
    def get_batch(self, batch_size, n_steps, regression=True):
        rnd = (int) ((self.data.shape[0] - n_steps - batch_size + 1) * np.random.rand())
        batch_x = self.data[rnd:rnd + n_steps].reshape(-1)
            
        for i in range(batch_size - 1):
            rnd += 1
            batch_x = np.hstack((batch_x, self.data[rnd:rnd + n_steps].reshape(-1)))

        batch_x = batch_x.reshape(batch_size, n_steps * 4)
   
        return batch_x
    
    def get_last_batch(self, batch_size, n_steps, regression=True):
        rnd = (int) (self.data.shape[0] - n_steps - batch_size + 1)
        batch_x = self.data[rnd:rnd + n_steps].reshape(-1)
            
        for i in range(batch_size - 1):
            rnd += 1
            batch_x = np.hstack((batch_x, self.data[rnd:rnd + n_steps].reshape(-1)))

        batch_x = batch_x.reshape(batch_size, n_steps * 4)
            
        return batch_x


# ## Module Building
# - Using sklearn.KNeighborsRegressor module
# 
# - Training Datas are newest 500 vectors
#  - $TrainingData = [v_{n-499}, v_{n-498}, ..., v_{n}]$
#  - where $v_i$ is the vector of the $i$-th day
# 
# - A vector contain prices of open-high-low-close in past 60 days
#  - $v_{n} = [o_{n-59}, h_{n-59}, l_{n-59}, c_{n-59}, o_{n-58}, h_{n-58}, l_{n-58}, c_{n-58}, ..., o_{n}, h_{n}, l_{n}, c_{n}]$
#  - where $o_i - h_i - l_i - c_i$ are open-high-low-close prices of the $i$-th day
# 
# - Target of training will predict an average of open prices in future 10 days
#  - $t_{n} = \frac{1}{10}\sum_{i=n+1}^{n+11} o_i$
#  
# - We'll be re-training when each testing data is inputted

# In[ ]:


class Scikit_KNeighborsRegressor:
    def __init__(self, data, length=60, interval=10, batch=500):
        self.module = None
        self.length = length
        self.first = True
        self.iterator = 0
        self.test_data = np.zeros((1, length * 4))
        self.stock = 0
        self.batch = batch
        self.interval = interval
        self.store_data = np.zeros((1, (length + interval) * 4))
        self.retrain_data = np.zeros((1, length * 4))
        self.retrain_target = 0
        self.dataset = Datasets(data, interval=interval)
    
    def _insert_data(self, row):
        if self.first:
            for i in range(self.dataset.data.shape[0] - self.length, self.dataset.data.shape[0]):
                self.test_data[0] = np.hstack((self.dataset.data[i], self.test_data[0, :(self.length - 1) * 4]))
            self.test_data[0] = np.hstack((row, self.test_data[0, :(self.length - 1) * 4]))

            for i in range(self.length + self.interval):
                self.store_data[0, i * 4:i * 4 + 4] = row
            self.first = False
        else:
            self.test_data[0] = np.hstack((row, self.test_data[0, :(self.length - 1) * 4]))
            self.store_data[0] = np.hstack((row, self.store_data[0, :(self.length + self.interval - 1) * 4]))
        
        self.iterator += 1
        
        self.retrain_data = self.store_data[:, self.interval * 4:]
        self.retrain_target = [np.mean(self.store_data[0, ::4][:self.interval])]
        
    def train(self):
        data_x = self.dataset.get_last_batch(5 * self.length, self.length)
        data_t = self.dataset.target_regression[self.dataset.target_regression.shape[0] - 5 * self.length:]

        train_x, test_x, train_t, test_t = train_test_split(data_x, data_t, random_state=4)
        
        self.module = KNeighborsRegressor()
        self.module.fit(train_x, train_t)
        
    def predict_action(self, row):
        self.dataset.add_data(row)
        self._insert_data(row)
        pred = self.module.predict(self.test_data)
        trade = 0
        
        if pred[0] > 3 :
            ref = 1
            if self.stock != 1:
                trade = 1
        elif pred[0] < -3:
            ref = -1
            if self.stock != -1:
                trade = -1
        else:
            ref = 0
        
        self.stock += trade
        return trade
    
    def re_training(self):
        if self.batch <= self.dataset.target_regression.shape[0] - self.length + 1:
            batch = self.batch
        else:
            batch = self.dataset.target_regression.shape[0] - self.length + 1
            
        data_x = self.dataset.get_last_batch(self.batch, self.length)
        data_t = self.dataset.target_regression[self.dataset.target_regression.shape[0] - self.batch:]
    
        train_x, test_x, train_t, test_t = train_test_split(data_x, data_t, random_state=4)
        
        self.module = KNeighborsRegressor()
        self.module.fit(train_x, train_t)
        


# ## AutoTrader implement

# In[ ]:


class Trader:
    def __init__(self, batch_size=1, n_steps=60):
        self.dataset = None
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.module = None
        
    def train(self, data):
        self.module = Scikit_KNeighborsRegressor(data)
        self.module.train()

    def predict_action(self, row):
        return str(self.module.predict_action(row)) + '\n'
    
    def re_training(self):
        self.module.re_training()
        


# ## Main module

# In[ ]:


# You can write code above the if-main block.

if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    parser.add_argument('-f',
                        default='',
                        help='ipython')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    trader = Trader()
    trader.train(training_data)
    
    testing_data = load_data(args.testing)
    
    # Show result
    buy_and_hold_strategy = testing_data[testing_data.shape[0] - 1, 3] + testing_data[testing_data.shape[0] - 1, 0] - testing_data[1,0]
    
    i = 1
    stock = 0
    action = 0
    money = 0
    # print('i', 'action', 'stock', 'money')
    
    with open(args.output, 'w') as output_file:
        for row in testing_data:
            # We will perform your action as the open price in the next day.
            money -= int(action) * row[0]
            action = trader.predict_action(row)

            # Show result
            # print(i, int(action), stock, round(money, 4))
            stock += int(action)
            i += 1

            if i <= testing_data.shape[0]:
                output_file.write(action)

            # this is your option, you can leave it empty.
            trader.re_training()

    # Show result
    money += stock * (testing_data[testing_data.shape[0] - 1, 3] + testing_data[testing_data.shape[0] - 1, 0])
    print('total money', round(money, 4))
    print('buy_and_hold_strategy', round(buy_and_hold_strategy, 4))
    print('batter', round((money - buy_and_hold_strategy) * 100 / buy_and_hold_strategy, 4), '%')
        


# ## Conclusion
# - We use the newest 800 datas for training because maybe the overly older data will make misunderstand
# - The open price is unsettled in short time, so predict the open price in next day will be difficult and inaccurate
# 
