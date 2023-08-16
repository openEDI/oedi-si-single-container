import numpy as np
import torch
from torch import nn
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class disaggrData(Dataset):
    def __init__(self, x, length_input=48):
        super(disaggrData, self).__init__()
        self.res = x['res']
        self.solar = x['solar']
        self.aggr = x['aggr']
        self.inLen = length_input

    def __getitem__(self, index):

        X = np.array(self.aggr[index]).reshape((48, 1))
        Y1 = np.array(self.res[index]).reshape((48, 1))
        Y2 = np.array(self.solar[index]).reshape((48, 1))

        return X, Y1, Y2

    def __len__(self):
        a=len(self.aggr)
        return len(self.aggr)


class DNN_ld(nn.Module):
    def __init__(self, hidden_size, layer_num):
        super(DNN_ld, self).__init__()
        self.fc_final = nn.Sequential(
            nn.Linear(48, 128),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 48),)

        self.fc_final2 = nn.Sequential(
            nn.Linear(48, 128),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 48), )

    def forward(self, input_data):
        #self.lstm_layer.flatten_parameters()
        #input_encoded, lstm_states = self.lstm_layer(input_data)
        #input_data1 = input_data.view(-1, 96)
        input_data=input_data.squeeze()
        y1 = self.fc_final(input_data)
        y2 = self.fc_final2(input_data)
        return y1,y2


def evaluate_score(y_real, y_predict):
    # MAE
    print('MAE', metrics.mean_absolute_error(y_real, y_predict))


if __name__ == '__main__':
    epochs = 100
    input_length = 48
    learning_rate = 0.01
    hidden = 64
    layer = 1
   # train = np.load('data/train2.npy', allow_pickle=True)
   # val = np.load('data/val2.npy', allow_pickle=True)
    #length = len(val) - input_length + 1

    with open('train_aus.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('val_aus.pkl', 'rb') as f2:
        val = pickle.load(f2)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train1={}
    val1={}

   # for x,y in train.items():
   #     train_scaled = scaler.fit_transform(y)
   #     train1[x]= train_scaled

   # for x1,y1 in val.items():
   #     val_scaled = scaler.fit_transform(y1)
   #     val1[x1]= val_scaled

    train1=train
    val1=val

    trainData = disaggrData(train1, input_length)
    valiData = disaggrData(val1, input_length)

    train_Dataloader = DataLoader(
        trainData,
        batch_size=60,
        shuffle=True)

    vali_Dataloader = DataLoader(
        valiData,
        batch_size=1,
        #sampler=range(0, length, input_length)
    )


    model = DNN_ld(hidden_size=hidden, layer_num=layer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Training start")
    for e_i in range(epochs):
        print(f"# of epoches: {e_i}")
        for t_i, (X, Y1, Y2) in enumerate(train_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            Y1 = Y1.type(torch.FloatTensor).to(device)
            Y2 = Y2.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            y_pred1,y_pred2 = model(X)

            # loss = criterion(y_pred[:,:,:1]+y_pred[:,:,-1:], X)+criterion(y_pred[:,:,:1], Y1) + criterion(y_pred[:,:,-1:], Y2)
            loss = criterion(y_pred1, Y1.squeeze()) + criterion(y_pred2, Y2.squeeze())
            loss.backward()
            optimizer.step()
    print("Training end")

    print("Validation start")

    y_pred = []
    y1_pred=[]
    y2_pred=[]
    XX = []
    with torch.no_grad():
        for _, (X, Y1, Y2) in enumerate(vali_Dataloader):
            X = X.type(torch.FloatTensor).to(device)
            Y1 = Y1.type(torch.FloatTensor).to(device)
            Y2 = Y2.type(torch.FloatTensor).to(device)
            output1,output2 = model(X)
            #y_pred.append(output1)
            y1_pred.append(output1.squeeze().cpu().numpy())
            y2_pred.append(output2.squeeze().cpu().numpy())
            XX.append(X.squeeze(0))
        #final_y_pred = torch.cat(y_pred, 0).cpu()
        #final_x = torch.cat(XX, 0).cpu()

    #final_scaled = torch.cat((final_x, final_y_pred),1)

   # final_y1 = scaler.inverse_transform(y1_pred)
   # final_y2 = scaler.inverse_transform(y2_pred)

    final_y1 = y1_pred
    final_y2 = y2_pred

    agg_real = val['aggr']
    solar_real = val['solar']
    solar_pred = final_y2

    res_real = val['res']
    res_pred = final_y1

    evaluate_score(solar_real, solar_pred)

    evaluate_score(res_real, res_pred)

    score1 = norm(np.array(res_pred) - np.array(res_real), 'fro') / norm(np.array(res_real), 'fro')
    score2 = norm(np.array(solar_pred) - np.array(solar_real), 'fro') / norm(np.array(solar_real), 'fro')

    print("score1=", score1)
    print("score2=", score2)

    index_d = 6;
    plt.figure()
    plt.plot(solar_real[index_d], label='Ground Truth Solar')
    plt.plot(solar_pred[index_d], label="Estimated Solar")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(res_real[index_d], label='Ground Truth Res')
    plt.plot(res_pred[index_d], label="Estimated Residential")
    plt.legend()
    plt.show()

    csfont = {'fontname': 'Times New Roman'}

    t1 = np.arange(0.0, 48.0, 2)
    t1 = range(48)
    plt.figure(figsize=(8, 6))
    plt.subplot(311)
    plt.plot(t1, agg_real[index_d], linewidth=2.0, label='Net Load', color='b', alpha=0.6)
    plt.legend()
    # plt.xlabel('Time',fontsize=14,**csfont)
    plt.ylabel('Active Power (MW)', fontsize=12, **csfont)
    # plt.title('Net Load',fontsize=14, **csfont)
    plt.xticks([0, 8 - 1, 16 - 1, 24 - 1, 32 - 1, 40 - 1], ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00'],
               fontsize=12, )
    plt.yticks(fontsize=12)

    plt.subplot(312)
    plt.plot(t1, res_pred[index_d], linewidth=2.0, label='Estimated load', color='g', alpha=0.6)
    plt.plot(t1, res_real[index_d], linewidth=2.0, label='Ground Truth load', color='r', alpha=0.6)
    plt.legend()
    # plt.xlabel('Time',fontsize=14,**csfont)
    plt.ylabel('Active Power (MW)', fontsize=12, **csfont)
    # plt.title('Disaggregation Results',fontsize=14, **csfont)
    plt.xticks([0, 8 - 1, 16 - 1, 24 - 1, 32 - 1, 40 - 1], ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00'],
               fontsize=12, )
    plt.yticks(fontsize=12)

    # lt.show(311)

    plt.subplot(313)
    plt.plot(t1, -solar_real[index_d], linewidth=2.0, label='Estimated load', color='g', alpha=0.6)
    plt.plot(t1, -solar_pred[index_d], linewidth=2.0, label='Ground Truth load', color='r', alpha=0.6)
    plt.legend()
    plt.xlabel('Time', fontsize=14, **csfont)
    plt.ylabel('Active Power (MW)', fontsize=12, **csfont)
    # plt.title('Disaggregation Results',fontsize=14, **csfont)
    plt.xticks([0, 8 - 1, 16 - 1, 24 - 1, 32 - 1, 40 - 1], ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00'],
               fontsize=12, )
    plt.yticks(fontsize=12)
    plt.savefig('figureDNN.png')
    plt.show()

    error1 = norm(res_pred[index_d] - res_real[index_d], 2) / norm(res_real[index_d], 2)
    error2 = norm(solar_pred[index_d] - solar_real[index_d]) / norm(solar_real[index_d], 2)

    print("error1=", error1)
    print("error2=", error2)

    print("Validation end")
