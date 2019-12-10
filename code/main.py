from torch import nn, optim
from prepareData import prepare_data
from model import Model
from trainData import Dataset


class Config(object):
    def __init__(self):
        self.data_path = '../data'
        self.validation = 5
        self.save_path = '../data'
        self.epoch = 300
        self.alpha = 0.2


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()


class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32


def train(model, train_data, optimizer, opt):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), score)
        loss.backward()
        optimizer.step()
        return loss
    for epoch in range(1, opt.epoch+1):
        train_reg_loss = train_epoch()
        print(train_reg_loss.item()/(len(one_index[0])+len(zero_index[0])))


opt = Config()


def main():
    dataset = prepare_data(opt)
    sizes = Sizes(dataset)
    train_data = Dataset(opt, dataset)
    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_data[i], optimizer, opt)


if __name__ == "__main__":
    main()
