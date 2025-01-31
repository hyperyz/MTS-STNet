import json
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from tools.data import get_dataloader
from tools.utils import log_to_file, make_saved_dir
from model.mts import mts
from tools.metrics import MAE_torch, RMSE_torch, MAPE_torch


class Trainer:
    def __init__(self, json_file):
        print('Loading...')
        with open(json_file, 'r') as file:
            conf = json.load(file)

        self.device, self.epochs = conf["device"], conf["epochs"]
        self.saved_dir = make_saved_dir(conf["saved_dir"])
        loaders = get_dataloader(conf["data_file"], conf["batch_size"], conf["device"])
        self.train_log = partial(log_to_file, f'{self.saved_dir}/train.log')
        self.validate_log = partial(log_to_file, f'{self.saved_dir}/validate.log')
        self.train_loader, self.validate_loader, self.test_loader, self.scaler = loaders
        # torch.save(statistics, f'{self.saved_dir}/statistics.pth')

        num_nodes = conf["n_nodes"]
        self.model = mts(num_node=num_nodes, input_dim=1)
        self.model.to(conf["device"])

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=conf["lr"], eps=1.0e-8, weight_decay=0,
                                          amsgrad=False)
        self.criterion = torch.nn.L1Loss().to(conf["device"])
        self.patience = 20

    def fit(self):
        print(f'Training...')
        # train
        best = float('inf')
        best_model = None
        history = []
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}")

            t_loss = self.train_epoch(epoch)
            v_loss, mae = self.validate_epoch(epoch)
            if mae < best:
                best = mae
                best_model = self.model.state_dict()
                torch.save(best_model, f'{self.saved_dir}/model-{best:.2f}.pkl')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f'Early stopping at epoch {epoch + 1} with best MAE: {best:.2f}')
                torch.save(best_model, f'{self.saved_dir}/model-{best:.2f}.pkl')
                break
            history.append(dict(train_loss=t_loss, validate_loss=v_loss, metrics={'MAE': mae}))
        open(f'{self.saved_dir}/history.json', 'w').write(json.dumps(history))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, average_loss = .0, .0
        with tqdm(total=len(self.train_loader), desc='Training', unit='batch') as bar:
            for idx, (x, y) in enumerate(self.train_loader):
                y = self.scaler.inverse_transform(y)
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)

                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                average_loss = total_loss / (idx + 1)
                bar.set_postfix(loss=f'{average_loss:.2f}')
                bar.update(1)
                self.train_log(epoch=epoch, batch=idx, loss=loss)

        return average_loss

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss, average_loss = .0, .0
        total_mae = 0
        total_rmse = 0
        total_mape = 0

        with tqdm(total=len(self.validate_loader), desc='Validating', unit='batch') as bar:
            for idx, (x, y) in enumerate(self.validate_loader):
                y = self.scaler.inverse_transform(y)
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)
                pred = pred.contiguous()
                y = y.contiguous()

                total_mae += MAE_torch(pred, y)
                total_rmse += RMSE_torch(pred, y)
                total_mape += MAPE_torch(pred, y, 1)

                total_loss += loss.item()
                average_loss = total_loss / (idx + 1)

                bar.set_postfix(loss=f'{average_loss:.2f}')
                bar.update(1)
                self.validate_log(epoch=epoch, batch=idx, loss=loss)

        avg_mae = total_mae / len(self.validate_loader)
        avg_rmse = total_rmse / len(self.validate_loader)
        avg_mape = total_mape / len(self.validate_loader)
        print("Validation MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%".format(avg_mae, avg_rmse, avg_mape * 100))
        return average_loss, avg_mae.item()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'Model loaded from {model_path}')
        return self.model

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        predictions = []
        true_values = []
        total_mae = 0
        total_rmse = 0
        total_mape = 0

        with tqdm(total=len(self.test_loader), desc='Predicting', unit='batch') as bar:
            for batch in self.test_loader:
                x, y = (it.to(self.device) for it in batch)
                y = self.scaler.inverse_transform(y)
                pred = self.model(x)

                total_mae += MAE_torch(pred, y)
                total_rmse += RMSE_torch(pred, y)
                total_mape += MAPE_torch(pred, y, 1)

                predictions.append(pred.cpu().numpy())
                true_values.append(y.cpu().numpy())

                bar.update(1)

        avg_mae = total_mae / len(self.test_loader)
        avg_rmse = total_rmse / len(self.test_loader)
        avg_mape = total_mape / len(self.test_loader)

        print("MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%".format(avg_mae, avg_rmse, avg_mape * 100))
        return predictions, true_values, avg_mae.item(), avg_rmse.item(), avg_mape.item()
