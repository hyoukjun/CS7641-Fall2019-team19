import numpy as np
import torch
import math
import csv
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import seaborn as sns

import matplotlib.pyplot as plt
import imageio


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso


def rmse(pred, label): 
    '''
    This is the root mean square error.
    Args:
        pred: numpy array of length N * 1, the prediction of labels
        label: numpy array of length N * 1, the ground truth of labels
    Return:
        a float value
    '''
    rmse_vals = np.sqrt(np.mean((pred-label)**2))
    return rmse_vals

def abs_avg_err(pred, label):
    err = np.mean(abs(pred-label))
    return err

class LinearReg(object):
    def fit_closed(xtrain, ytrain):
        XT = xtrain.T
        XTX = np.matmul(XT, xtrain)
        XTX_inv = np.linalg.inv(XTX)
        ret = np.matmul(np.matmul(XTX_inv, XT), ytrain)
        return ret             

    def predict(xtest, weight):
        prediction = np.matmul(xtest, weight)
        
        return prediction

def cross_validation_lin(X, y, kfold):
    N = X.shape[0]
    foldsize = N/kfold
    foldsize = int(foldsize)
    
    rmse_val = 0
    
    for fold in range(0, kfold):
        base_idx = fold * foldsize
        bound_idx = (fold + 1) * foldsize-1
        
        xtrain = np.concatenate((X[0:base_idx,:], X[bound_idx+1:N-1,:]), axis=0)
        ytrain = np.concatenate((y[0:base_idx], y[bound_idx+1:N-1]), axis=0)
        weight = LinearReg.fit_closed(xtrain, ytrain)        

        
        xtest = X[base_idx:bound_idx+1,:]
        ytest = y[base_idx:bound_idx+1]        
        ypred = LinearReg.predict(xtest, weight)
                
        rmse_val = rmse_val + rmse(ypred, ytest)
    
    rmse_val = rmse_val / 10
    
    return rmse_val
        

class RidgeReg(LinearReg):
    def fit_closed(xtrain, ytrain, c_lambda):
        
        N = xtrain.shape[0]
        D = xtrain.shape[1]
        
        pinv_1 = np.linalg.inv(np.add(np.matmul(xtrain.T, xtrain), np.multiply(c_lambda, np.identity(D))))
        ret = np.matmul(np.matmul(pinv_1, xtrain.T), ytrain)
        return ret

def cross_validation_rid(X, y, kfold, c_lambda):
    N = X.shape[0]
    foldsize = N/kfold
    foldsize = int(foldsize)
    
    rmse_val = 0

    for fold in range(0, kfold):
        base_idx = fold * foldsize
        bound_idx = (fold + 1) * foldsize-1
        
        xtrain = np.concatenate((X[0:base_idx,:], X[bound_idx+1:N-1,:]), axis=0)
        ytrain = np.concatenate((y[0:base_idx], y[bound_idx+1:N-1]), axis=0)
        weight = RidgeReg.fit_closed(xtrain, ytrain, c_lambda)        

        
        xtest = X[base_idx:bound_idx+1,:]
        ytest = y[base_idx:bound_idx+1]        
        ypred = RidgeReg.predict(xtest, weight)
                
        rmse_val = rmse_val + rmse(ypred, ytest)
    
    rmse_val = rmse_val / kfold
    
    return rmse_val

def cross_validation_log(X, y, kfold):
    N = X.shape[0]
    foldsize = N/kfold
    foldsize = int(foldsize)
    
    rmse_val = 0
    
    for fold in range(0, kfold):
        base_idx = fold * foldsize
        bound_idx = (fold + 1) * foldsize-1
        
        xtrain = np.concatenate((X[0:base_idx,:], X[bound_idx+1:N-1,:]), axis=0)
        ytrain = np.concatenate((y[0:base_idx], y[bound_idx+1:N-1]), axis=0)

        xtest = X[base_idx:bound_idx+1,:]
        ytest = y[base_idx:bound_idx+1]

        logistic_regression = LogisticRegression(solver='sag',n_jobs=4, max_iter=1000, tol=1, multi_class='auto')
        logistic_regression.fit(xtrain,ytrain)
        ypred = logistic_regression.predict(xtest)

        rmse_val = rmse_val + rmse(ypred, ytest)
    
    rmse_val = rmse_val / kfold
    
    return rmse_val

def cross_validation_Lasso(X, y, kfold, alpha):
    N = X.shape[0]
    foldsize = N/kfold
    foldsize = int(foldsize)
    
    rmse_val = 0
    
    for fold in range(0, kfold):
        base_idx = fold * foldsize
        bound_idx = (fold + 1) * foldsize-1
        
        xtrain = np.concatenate((X[0:base_idx,:], X[bound_idx+1:N-1,:]), axis=0)
        ytrain = np.concatenate((y[0:base_idx], y[bound_idx+1:N-1]), axis=0)

        xtest = X[base_idx:bound_idx+1,:]
        ytest = y[base_idx:bound_idx+1]

        lasso_regression = Lasso(alpha=alpha,normalize=True, max_iter=1e4, tol=1)
        lasso_regression.fit(xtrain,ytrain)
        ypred = lasso_regression.predict(xtest)

        rmse_val = rmse_val + rmse(ypred, ytest)
    
    rmse_val = rmse_val / kfold
    
    return rmse_val

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
#            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, 1)
            self.relu3 = torch.nn.ReLU()            
            self.fc4 = torch.nn.Linear(self.hidden_size, 1)
            self.relu4 = torch.nn.ReLU()            
            self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu5 = torch.nn.ReLU()            
            self.fc6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu6 = torch.nn.ReLU()            
            self.fc7 = torch.nn.Linear(self.hidden_size, 1)
            self.relu7 = torch.nn.ReLU()            
            self.fc8 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu8 = torch.nn.ReLU()            
            self.fc9 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu9 = torch.nn.ReLU()            
            self.fc10 = torch.nn.Linear(self.hidden_size, 1)
            self.relu10 = torch.nn.ReLU()            
#            self.fc5 = torch.nn.Linear(self.hidden_size, 1)
#            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden1 = self.fc1(x)
            hidden1 = self.relu1(hidden1)
            hidden2 = self.fc2(hidden1)
            hidden2 = self.relu2(hidden2)
#            hidden2 = self.bn2(hidden2)
            hidden3 = self.fc3(hidden2)
#            hidden3 = self.relu3(hidden3)
#            hidden4 = self.fc4(hidden3)
#            hidden4 = self.relu4(hidden4)
#            hidden5 = self.fc5(hidden4)
#            hidden5 = self.relu5(hidden5)
#            hidden6 = self.fc6(hidden5)
#            hidden6 = self.relu6(hidden6)
#            hidden7 = self.fc7(hidden6)
#            hidden7 = self.relu7(hidden7)
#            hidden8 = self.fc8(hidden7)
#            hidden8 = self.relu8(hidden8)
#            hidden9 = self.fc9(hidden8)
#            hidden9 = self.relu9(hidden9)
#            hidden10 = self.fc10(hidden9)
#            hidden10 = self.relu10(hidden10)                        
            output = hidden3
            '''
            hidden2 = self.fc2(hidden1)
            hidden3 = self.fc3(hidden2)
            hidden4 = self.fc4(hidden3)
            hidden5 = self.fc5(hidden4)                        
            output = hidden5
            '''
#            relu1 = self.relu1(hidden1)
#            hidden2 = self.fc2(relu1)
#            relu2 = self.relu2(hidden2)
#            hidden3 = self.fc2(relu2)
#            relu3 = self.relu2(hidden3)
#            hidden4 = self.fc2(relu3)
#            relu4 = self.relu2(hidden4)
#            output = self.fc5(relu4)
#            output = self.sigmoid(output)
            return output

def cross_validation_mlp(X, y, kfold, num_epochs, hs = 20, c_lambda = 0.5, learning_rate = 0.001):
    N = X.shape[0]
    F = X.shape[1]
    foldsize = N/kfold
    foldsize = int(foldsize)

    lowest_loss = 10000000;
    avg_loss = 0;
    
    for fold in range(0, kfold):
        base_idx = fold * foldsize
        bound_idx = (fold + 1) * foldsize-1
        
        xtrain = Variable(torch.from_numpy(np.concatenate((X[0:base_idx,:], X[bound_idx+1:N-1,:]), axis=0)))
        ytrain = Variable(torch.from_numpy(np.concatenate((y[0:base_idx], y[bound_idx+1:N-1]), axis=0)))

        xtrain = xtrain.float()
        ytrain = ytrain.float()

        xtest = Variable(torch.from_numpy(X[base_idx:bound_idx+1,:]))
        ytest = Variable(torch.from_numpy(y[base_idx:bound_idx+1]))

        xtest = xtest.float()
        ytest = ytest.float()

        mlp_model = Feedforward(F, hs)
        criterion = torch.nn.MSELoss()
#        criterion = torch.nn.MSELoss(reduction='mean')        
        mse_loss_func = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = learning_rate, weight_decay=10)
        mlp_model.eval()
        mlp_model = mlp_model.float()
        ypred = mlp_model(xtest)
        before_train = mse_loss_func(ypred.squeeze(), ytest)
        print('Test loss before training' , math.sqrt(abs(before_train.item())))

        mlp_plot_data = np.zeros(num_epochs)

        mlp_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Forward pass
            ypred = mlp_model(xtrain)
            # Compute Loss
            loss = criterion(ypred.squeeze(), ytrain)
            mse_loss = mse_loss_func(ypred.squeeze(), ytrain)

            c_lambda_tensor = torch.tensor(c_lambda)
            l2_reg = torch.tensor(0.)

            for param in mlp_model.parameters():
                l2_reg += torch.norm(param)
            loss += c_lambda_tensor * l2_reg

            if fold == 0:
                mlp_plot_data[epoch] = math.sqrt(mse_loss.item())

            if epoch % 100 == 0:
                print('Epoch {}: train loss: {}'.format(epoch, math.sqrt(mse_loss.item())))

            # Backward pass
            loss.backward()
            optimizer.step()

        if fold == 0:
            D = mlp_plot_data.shape[0]
            plt_x = np.arange(D)
            dp_line_plot_plain(plt_x, mlp_plot_data, 'MLP', 'Epoch', 'RMSE')
            plt.savefig('mlp_epoch.png')
            plt.clf()

        ypred = mlp_model(xtest)
        after_train = mse_loss_func(ypred.squeeze(), ytest)
        print('Test loss after training' , math.sqrt(abs(after_train.item())))

        num_vals = N


        avg_loss = avg_loss + math.sqrt(abs(after_train.item()))
        
        if math.sqrt(abs(after_train.item())) < lowest_loss:
            lowest_loss = math.sqrt(abs(after_train.item()))
    
    avg_loss = avg_loss / kfold            

    return lowest_loss, avg_loss


def reject_outliers(data, m=2):
    return data[abs(data[:,22] - np.mean(data[:,22])) < m * np.std(data[:,22])]

def plot_curve(x, y, curve_type='.', color='b', lw=2):
    plt.plot(x, y, curve_type, color=color, linewidth=lw)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

def scree_plot(S, n_components=10):
    D = S.shape[0]
    
    sum_var = np.sum(S,axis=0)
    proportion_var = np.divide(S, sum_var)
    
    x = np.arange(D)
    plt.plot(x, proportion_var)    

def dp_line_plot(x_labels, S, title_text='', x_axis_label= '', y_axis_label = ''):
    D = S.shape[0]
    x = np.arange(D)
    plt.title(title_text)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xticks(x, x_labels)    
    plt.plot(x, S)


def dp_line_plot_plain(x_labels, S, title_text='', x_axis_label= '', y_axis_label = ''):
    D = S.shape[0]
    x = np.arange(D)
    plt.title(title_text)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x, S)


dataset = np.genfromtxt('inputData.csv', delimiter=',',dtype=float)

prices = dataset[:,22]

plt.title('Price Overview')
plt.xlabel('Datapoint ID')
plt.ylabel('Price ($)')
plt.plot(prices)
plt.savefig('prices.png')
plt.clf() 

dataset = reject_outliers(dataset, 1)

N = dataset.shape[0]
D = 23

features = dataset[:,1:21]
labels = dataset[:,22]
price_range = dataset[:,23]


feature_list = ['NeighborhoodGroup', 'Neighborhood', 'latitude', 'longitude', 'RoomType',
'minimum_nights', 'number_of_reviews', 'DaysFromLastReview', 'ReviewsPerMonth', 'host_listings_count',
'availability_365', 'Amenities', 'Cost of living', 'Crime', 'Employment', 'Housing', 'School', 'Weather', 
'home_median', 'rent_median', 'livability', 'Price']

corrset = dataset[:,1:22]

corr = np.corrcoef(corrset, labels, rowvar=False)

plt.figure(figsize=(42, 47))
sns.set(font_scale=3)

ax = sns.heatmap(corr, xticklabels = feature_list, yticklabels = feature_list)

D = corr.shape[0]
x = np.arange(D)
plt.title('Correlation of Features')
plt.xlabel('Feature ID')
plt.ylabel('Feature ID')
plt.savefig('correlation.png')
plt.clf()



mean_rmse = cross_validation_lin(features, labels, 10)

print("Linear Regression Mean RMSE: ", mean_rmse)


best_lambda = None
best_error = None
kfold = 10
lambda_list = [0, 0.1, 1, 5, 10, 100, 1000]

rid_plot_data = np.zeros(7)

count = 0
for lm in lambda_list:
    err = cross_validation_rid(features, labels, kfold, lm)
    rid_plot_data[count] = err
    count = count+1
    print('lambda: %.2f' % lm, 'error: %.6f'% err)
    if best_error is None or err < best_error:
        best_error = err
        best_lambda = lm


dp_line_plot(lambda_list, rid_plot_data, 'Ridge Regression', 'Lambda', 'RMSE')
plt.savefig('ridge.png')

plt.clf()

print("Ridge Regression Mean RMSE: ", best_error, " (with lambda = ", best_lambda, ")")


mean_rmse = cross_validation_log(features, labels, 10)
print("Logistic Regression Mean RMSE: ", mean_rmse)

best_alpha = None
best_error = None
kfold = 10
alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]

lasso_plot_data = np.zeros(7)

count = 0
for alp in alpha_list:
    err = cross_validation_Lasso(features, labels, kfold, alp)
    lasso_plot_data[count] = err
    count = count + 1
    print('alpha: %.4f' % alp, 'error: %.6f'% err)
    if best_error is None or err < best_error:
        best_error = err
        best_alpha = alp

dp_line_plot(alpha_list, lasso_plot_data, 'Lasso Regression', 'Alpha', 'RMSE')
plt.savefig('lasso.png')

plt.clf()


print("Lasso Regression Mean RMSE: ", best_error, " (with alpha = ", best_alpha, ")")


hs_list = [10, 20, 30, 40, 50]
mlp_plot_data = np.zeros(5)

lowest_loss = 100000
avg_loss = 0
count = 0
for hs in hs_list:
    print("Hidden layer size: ", hs)
    lowest_loss, avg_loss = cross_validation_mlp(features, labels, 10, 1000, hs, 0.5, 0.005)
    mlp_plot_data[count] = avg_loss
    count = count + 1
    print("lowest MLP loss: ", lowest_loss)
    print("avg MLP loss: ", avg_loss)


dp_line_plot(hs_list, mlp_plot_data, 'MLP - Impact of Hidden Layer Size', 'Hs', 'RMSE')
plt.savefig('MLP_Hs.png')

plt.clf()

