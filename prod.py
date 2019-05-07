# -*- coding: utf-8 -*-
import fastai
from fastai import *
from fastai.text import *
from functools import partial
import io
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets

learn_trained = load_learner(model_path)

def record_review(org_name, polarity_value):
  with open("test.txt", "a") as myfile:
    write_text = org_name+ ',' + str(polarity_value[0]) + ',' + str(polarity_value[1]) + '\n'
    myfile.write(write_text)
    
def overall_company_review(name_of_company):
  data = pd.read_csv('test.txt', names=['company','positive','negative'])
  value = data.groupby(['company']).mean().loc[name_of_company]
  sns.barplot(x=['positive','negative'], y=tuple(reversed([100*val for val in value])))
  print('overall opinion of employee for',name_of_company,'is positive',value[1],' negative',value[0])
  
# from matplotlib.widgets import RadioButtons

def find_result(x):
  if abs(x[0]-x[1]) <= 0.15:
    return ('neutral',5)
  elif (x[0] - x[1]) >= 0.6:
    return ('extremly negative',1)
  elif (x[1] - x[0]) >= 0.6:
    return ('extremly positive',9)
  elif x[0] > x[1]:
    return ('negative',3)
  else:
    return('positive',7)
def prediction_result(x):
  a= learn_trained.predict(x)
  size = a[2].tolist()
  labels = 'negative', 'positive'
  colors = ['yellowgreen','lightskyblue']
#   a = plt.pie(size, labels=labels, colors=colors,shadow=True,autopct='%1.0f%%')
#   radio = RadioButtons(a, ('extremly negative', 'negative', 'neutral','positive','extremly positive'))
#   plt.show()
#   print(size)
  print('Text:',x)
  result = find_result(size)
  print("Sentiment class:",result[0])
  return result[1],size
  
def review_show(org_name,review):
  res = prediction_result(review)
  widgets.IntSlider(value=res[0],min=0,max=10,step=1,description='Sentiment:',disabled=False,continuous_update=False,orientation='horizontal',readout=False,readout_format='d')
  record_review(org_name, res[1])


print(review_show('leapfrog','it feels good working in abc company'))
print(overall_company_review('leapfrog'))




