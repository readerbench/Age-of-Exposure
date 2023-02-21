import logging
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import (cross_val_score)
from sklearn.model_selection import train_test_split, KFold
from torch.utils import data

from features_loading_utils import *


logging.basicConfig(
    level=logging.DEBUG,
    format=('%(asctime)s '
            '%(levelname)s '
            '%(filename)s '
            '%(funcName)s() '
            '%(lineno)d --- \t'
            '%(message)s')
)
log = logging.getLogger('train_model_wv')

WV_FILE_SRC = '../data/word_vectors_fixed/en_map_fix_word_vectors_full.bin'
WV_FILE_TGT = '../data/word_vectors_fixed/de_map_fix_word_vectors_full.bin'
AOA_FILE_SRC = '../data/aoa_en.csv'
AOA_FILE_TGT = '../data/aoa_de.csv'
FEATURES_FILE_SRC = '../data/word_vectors_fixed/en_map_new_aoa_order_indices_20steps.csv'
FEATURES_FILE_TGT = '../data/word_vectors_fixed/de_map_new_aoa_order_indices_20steps.csv'
WORD_FREQ = '../data/word_freq/de'

WEIGHTS_TGT_BIN_CKPT = "weights_de.bin"
BEST_WEIGHTS_TGT_CKPT = 'best_weights_de.bin'
WEIGHTS_SRC_BIN_CKPT = "weights_en.bin"
BEST_SRC_CKPT_FILE = 'best_weights_en.bin'


class WordVectorDataset(torch.utils.data.Dataset):

    def __init__(self, wv_file, features_file, aoa_data):
        super().__init__()
        self.wv_file = wv_file
        self.data = self.load_data(aoa_data, features_file)

    def load_data(self, aoa_data, features_file):
        log.info("Loading data %s" % self.wv_file)
        with open(self.wv_file, 'rb') as fin:
            wv = pickle.load(fin)
            print("WV size:", len(wv))
        aoa_scores = aoa_data.set_index('Word').to_dict()['AoA']
        X, X_no_vif, _, words = get_data(aoa_scores, features_file, WORD_FREQ)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        data = []
        for i, word in enumerate(words):
            if word not in wv:
                continue
            embeddings = np.asarray(wv[word], np.float32)
            data.append({
                'word': word,
                'wv': torch.from_numpy(embeddings).float(),
                'score': torch.tensor(aoa_scores[word]).float(),
                'features': torch.from_numpy(X[i]).float()
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        return e['wv'], e['features'], e['score']


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, wv_file, features_file):
        super().__init__()
        self.wv_file = wv_file
        self.data = self.load_data(features_file)

    def load_data(self, features_file):
        log.info("Loading data %s" % self.wv_file)
        with open(self.wv_file, 'rb') as fin:
            wv = pickle.load(fin)
            print("WV size:", len(wv))
        X, X_no_vif, words = get_data_all(features_file, WORD_FREQ)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        data = []
        for i, word in enumerate(words):
            if word not in wv:
                continue
            embeddings = np.asarray(wv[word], np.float32)
            data.append({
                'word': word,
                'wv': torch.from_numpy(embeddings).float(),
                'score': torch.tensor(0).float(),
                'features': torch.from_numpy(X[i]).float()
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        return e['wv'], e['features'], e['score']


class AoeRegressor(nn.Module):

    def __init__(self, features_shape, emb_dim=300, hidden=16, layers=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.rnn_hidden = hidden * 2
        self.layers = layers

        self.rnn = nn.LSTM(self.emb_dim, self.rnn_hidden, layers, batch_first=True, bidirectional=True)
        self.rnn_fc = nn.Sequential(nn.Linear(self.rnn_hidden * 2, self.hidden), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(features_shape, self.hidden), nn.LeakyReLU())
        self.net = nn.Sequential(nn.Dropout1d(.2), nn.Linear(self.hidden * 2, 1))#, nn.ReLU())
        # self.net = nn.Sequential(
        #     nn.Linear(features_shape, self.hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden, 1),
        #     nn.ReLU()
        # )
        #
    def forward(self, x_rnn, x_features):
        h0 = torch.zeros(self.layers * 2, x_rnn.size(0), self.rnn_hidden, dtype=torch.float32)
        c0 = torch.zeros(self.layers * 2, x_rnn.size(0), self.rnn_hidden, dtype=torch.float32)
        out, (hn, cn) = self.rnn(x_rnn, (h0.detach(), c0.detach()))
        rnn_fc_out = self.rnn_fc(out[:, -1, :])
        fc_features_out = self.fc(x_features)
        out = torch.concat((rnn_fc_out, fc_features_out), -1)
        return self.net(out)
        # return self.net(x_features)


class AoeModule(pl.LightningModule):
    def __init__(self, model, loss, lr=1e-3, state_dict=None, load_weights=None):
        super().__init__()
        self.model = model
        self.lr = lr
        if load_weights is not None:
            self.model.load_state_dict(torch.load(load_weights))
        if state_dict is not None:
            self.model.load_state_dict({k[len('model.'):]: v for k, v in state_dict.items()})
        self.loss = loss

        self.metrics = {
            'train_mae': torchmetrics.MeanAbsoluteError(),
            'train_mse': torchmetrics.MeanSquaredError(),
            'val_mae': torchmetrics.MeanAbsoluteError(),
            'val_mse': torchmetrics.MeanSquaredError(),
            'train_r2': torchmetrics.R2Score(),
            'val_r2': torchmetrics.R2Score()
        }
        for metric in self.metrics:
            setattr(self, metric, self.metrics[metric])

    def training_step(self, batch, batch_idx):
        x, x_features, y = batch
        y_pred = self.model(x, x_features)
        loss = self.loss(y_pred, y.reshape(-1, 1))

        for metric in self.metrics:
            if metric.startswith('train_'):
                val = getattr(self, metric)(y_pred[:, 0], y)
                self.log(metric, val)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_features, y = batch
        y_pred = self.model(x, x_features)
        loss = self.loss(y_pred, y)

        for metric in self.metrics:
            if metric.startswith('val_'):
                val = getattr(self, metric)(y_pred[:, 0], y)
                self.log(metric, val)

        return loss

    def training_epoch_end(self, outs):
        for metric in self.metrics:
            if metric.startswith('train_'):
                log.info("%s: %.5f" % (metric, getattr(self, metric).compute()))
                self.log(metric, getattr(self, metric))

    def validation_epoch_end(self, outs):
        for metric in self.metrics:
            if metric.startswith('val_'):
                log.info("%s: %.5f" % (metric, getattr(self, metric).compute()))
                self.log(metric, getattr(self, metric))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def make_train_test(data_file, features_file, aoa_file):
    df = pd.read_csv(aoa_file)
    df['AoA'] = (df['AoA'] - df['AoA'].mean()) / df['AoA'].std()
    # df['AoA'] = df['AoA'] / df['AoA'].max()

    train, test = train_test_split(df, test_size=.2)
    train_dataset = WordVectorDataset(data_file, features_file, train)
    test_dataset = WordVectorDataset(data_file, features_file, test)
    return train_dataset, test_dataset


def make_all(data_file, features_file, aoa_file):
    df = pd.read_csv(aoa_file)
    df['AoA'] = (df['AoA'] - df['AoA'].mean()) / df['AoA'].std()
    # df['AoA'] = df['AoA'] / df['AoA'].max()

    return WordVectorDataset(data_file, features_file, df)


def evaluate_model(X, y, make_model):
    cv_mae = -cross_val_score(make_model(), X, y,
                              scoring='neg_mean_absolute_error',
                              cv=KFold(10, shuffle=True))
    print('MAE:', np.mean(cv_mae), 'norm:', np.mean(cv_mae) / np.max(y))
    print('R2:', np.mean(
        cross_val_score(make_model(), X, y, scoring='r2',
                        cv=KFold(10, shuffle=True))))


def get_best_model_score_from_ckpt(ckpt):
    return list(ckpt['callbacks'].values())[0]['best_model_score']


def train_simple():
    train_tgt, test_tgt = make_train_test(WV_FILE_TGT, FEATURES_FILE_TGT, AOA_FILE_TGT)
    model = AoeRegressor(len(train_tgt[0][1]))
    print(model)
    module = AoeModule(model, torch.nn.MSELoss())
    train_loader_tgt = data.DataLoader(train_tgt, shuffle=True, batch_size=32, num_workers=8)
    test_loader_tgt = data.DataLoader(test_tgt, shuffle=False, batch_size=32, num_workers=8)
    print(f"train_tgt={len(train_tgt)} test_tgt={len(test_tgt)}")

    if os.path.exists(WEIGHTS_TGT_BIN_CKPT):
        os.remove(WEIGHTS_TGT_BIN_CKPT)

    module = AoeModule(model, torch.nn.MSELoss())
    tgt_ckpt = ModelCheckpoint(dirpath='.', filename=WEIGHTS_TGT_BIN_CKPT, monitor='val_mse', save_top_k=1)
    trainer = pl.Trainer(max_epochs=7, logger=TensorBoardLogger('tb_logs_en_fr', name=f'en_fr_train_test'),
                         callbacks=[tgt_ckpt])
    trainer.fit(model=module, train_dataloaders=train_loader_tgt, val_dataloaders=test_loader_tgt)

    if not os.path.exists(BEST_WEIGHTS_TGT_CKPT + '.ckpt'):
        shutil.copy(WEIGHTS_TGT_BIN_CKPT + '.ckpt', BEST_WEIGHTS_TGT_CKPT + '.ckpt')
    best_ckpt = torch.load(BEST_WEIGHTS_TGT_CKPT + '.ckpt')
    best_model_score = get_best_model_score_from_ckpt(best_ckpt)
    if tgt_ckpt.best_model_score >= get_best_model_score_from_ckpt(best_ckpt):
        log.info(f"Will not update best model: "
                 f"current={tgt_ckpt.best_model_score} best={best_model_score}")
    else:
        log.info(f"Updating best model: current={tgt_ckpt.best_model_score} best={best_model_score}")
        os.replace(WEIGHTS_TGT_BIN_CKPT + '.ckpt', BEST_WEIGHTS_TGT_CKPT + '.ckpt')


def train_transfer():
    train_src, test_src = make_train_test(WV_FILE_SRC, FEATURES_FILE_SRC, AOA_FILE_SRC)
    train_tgt, test_tgt = make_train_test(WV_FILE_TGT, FEATURES_FILE_TGT, AOA_FILE_TGT)
    model = AoeRegressor(len(train_src[0][1]))
    print(model)
    module = AoeModule(model, torch.nn.MSELoss())
    train_loader_src = data.DataLoader(train_src, shuffle=True, batch_size=32, num_workers=8)
    test_loader_src = data.DataLoader(test_src, shuffle=False, batch_size=32, num_workers=8)
    train_loader_tgt = data.DataLoader(train_tgt, shuffle=True, batch_size=32, num_workers=8)
    test_loader_tgt = data.DataLoader(test_tgt, shuffle=False, batch_size=32, num_workers=8)
    print(f"train_src={len(train_src)} test_src={len(test_src)} train_tgt={len(train_tgt)} test_tgt={len(test_tgt)}")
    # full_src_dataset = FullDataset(WV_FILE_SRC, FEATURES_FILE_SRC)
    # full_tgt_dataset = FullDataset(WV_FILE_TGT, FEATURES_FILE_TGT)
    # full_src = data.DataLoader(full_src_dataset, shuffle=False, batch_size=32, num_workers=8)
    # full_tgt = data.DataLoader(full_tgt_dataset, shuffle=False, batch_size=32, num_workers=8)
    # print(f"full_src={len(full_src_dataset)} full_tgt={len(full_tgt_dataset)}")

    if os.path.exists(WEIGHTS_SRC_BIN_CKPT):
        os.remove(WEIGHTS_SRC_BIN_CKPT)
    if os.path.exists(WEIGHTS_TGT_BIN_CKPT):
        os.remove(WEIGHTS_TGT_BIN_CKPT)

    en_ckpt = ModelCheckpoint(dirpath='.', filename=WEIGHTS_SRC_BIN_CKPT, monitor='val_mse', save_top_k=1)
    trainer = pl.Trainer(max_epochs=7, logger=TensorBoardLogger('tb_logs_en_en', name=f'en_en_train_test'),
                         callbacks=[en_ckpt])
    trainer.fit(model=module, train_dataloaders=train_loader_src, val_dataloaders=test_loader_src)

    if not os.path.exists(BEST_SRC_CKPT_FILE + '.ckpt'):
        shutil.copy(WEIGHTS_SRC_BIN_CKPT + '.ckpt', BEST_SRC_CKPT_FILE + '.ckpt')
    best_ckpt = torch.load(BEST_SRC_CKPT_FILE + '.ckpt')
    best_model_score = get_best_model_score_from_ckpt(best_ckpt)
    if en_ckpt.best_model_score >= best_model_score:
        log.info(f"Will not update best model: "
                 f"current={en_ckpt.best_model_score} best={best_model_score}")
    else:
        log.info(f"Updating best model: current={en_ckpt.best_model_score} best={best_model_score}")
        os.replace(WEIGHTS_SRC_BIN_CKPT + '.ckpt', BEST_SRC_CKPT_FILE + '.ckpt')

    module = AoeModule(model, torch.nn.MSELoss(), state_dict=torch.load(BEST_SRC_CKPT_FILE + '.ckpt')['state_dict'])
    tgt_ckpt = ModelCheckpoint(dirpath='.', filename=WEIGHTS_TGT_BIN_CKPT, monitor='val_mse', save_top_k=1)
    trainer = pl.Trainer(max_epochs=7, logger=TensorBoardLogger('tb_logs_en_fr', name=f'en_fr_train_test'),
                         callbacks=[tgt_ckpt])
    trainer.fit(model=module, train_dataloaders=train_loader_tgt, val_dataloaders=test_loader_tgt)

    if not os.path.exists(BEST_WEIGHTS_TGT_CKPT + '.ckpt'):
        shutil.copy(WEIGHTS_TGT_BIN_CKPT + '.ckpt', BEST_WEIGHTS_TGT_CKPT + '.ckpt')
    best_ckpt = torch.load(BEST_WEIGHTS_TGT_CKPT + '.ckpt')
    best_model_score = get_best_model_score_from_ckpt(best_ckpt)
    if tgt_ckpt.best_model_score >= get_best_model_score_from_ckpt(best_ckpt):
        log.info(f"Will not update best model: "
                 f"current={tgt_ckpt.best_model_score} best={best_model_score}")
    else:
        log.info(f"Updating best model: current={tgt_ckpt.best_model_score} best={best_model_score}")
        os.replace(WEIGHTS_TGT_BIN_CKPT + '.ckpt', BEST_WEIGHTS_TGT_CKPT + '.ckpt')


def train_direct():
    train_src, test_src = make_train_test(WV_FILE_SRC, FEATURES_FILE_SRC, AOA_FILE_SRC)
    tgt = make_all(WV_FILE_TGT, FEATURES_FILE_TGT, AOA_FILE_TGT)
    model = AoeRegressor(len(train_src[0][1]))
    print(model)
    module = AoeModule(model, torch.nn.MSELoss())
    train_loader_src = data.DataLoader(train_src, shuffle=True, batch_size=32, num_workers=8)
    test_loader_src = data.DataLoader(test_src, shuffle=False, batch_size=32, num_workers=8)
    tgt_loader = data.DataLoader(tgt, shuffle=True, batch_size=32, num_workers=8)
    print(f"train_src={len(train_src)} test_src={len(test_src)} tgt={len(tgt)}")
    # full_src_dataset = FullDataset(WV_FILE_SRC, FEATURES_FILE_SRC)
    # full_tgt_dataset = FullDataset(WV_FILE_TGT, FEATURES_FILE_TGT)
    # full_src = data.DataLoader(full_src_dataset, shuffle=False, batch_size=32, num_workers=8)
    # full_tgt = data.DataLoader(full_tgt_dataset, shuffle=False, batch_size=32, num_workers=8)
    # print(f"full_src={len(full_src_dataset)} full_tgt={len(full_tgt_dataset)}")

    if not os.path.exists(BEST_SRC_CKPT_FILE + '.ckpt'):
        shutil.copy(WEIGHTS_SRC_BIN_CKPT + '.ckpt', BEST_SRC_CKPT_FILE + '.ckpt')

    ckpt = ModelCheckpoint(dirpath='.', filename=BEST_SRC_CKPT_FILE, monitor='val_mse', save_top_k=1)
    trainer = pl.Trainer(max_epochs=7, logger=TensorBoardLogger('tb_logs_en_en', name=f'en_en_train_test'), callbacks=[ckpt])
    trainer.fit(model=module, train_dataloaders=train_loader_src, val_dataloaders=[test_loader_src])
    trainer.validate(module, tgt_loader, ckpt_path=BEST_SRC_CKPT_FILE + '.ckpt')


if __name__ == '__main__':
    # train_simple()
    train_transfer()
    # train_direct()
