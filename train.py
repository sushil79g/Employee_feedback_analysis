import fastai
from fastai import *
from fastai.text import *
from functools import partial

import io
import os
import pandas as pd
from sklearn.utils import shuffle

data_lm = TextLMDataBunch.from_csv(csv_name='texts.csv',path='.',text_cols='text',label_cols='label')
data_clas = TextClasDataBunch.from_csv(csv_name='texts.csv',path='.',text_cols='text',label_cols='label', vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save('data_lm_export.pkl')
data_clas.save('data_clas_export.pkl')

data_lm = load_data('.', 'data_lm_export.pkl')
data_clas = load_data('.', 'data_clas_export.pkl', bs=16)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')

learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
