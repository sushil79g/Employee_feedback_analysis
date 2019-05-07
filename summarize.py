from tqdm import tqdm
from gensim.summarization.summarizer import summarize

def summerizer_overall(file, ratio = 0.1, col=None, attribute=None,review='review'):
  ''' 
    File represent the dataframe of dataset, Here col represent the features in dataset like: if the dataset is about the company review, Department can be the 
  the features in dataset, attribute represent sub-group of features like in above example if Department is 'col', attribute
  can the engineering depart or intern depart, review represent the feature where review of employee is located, ratio 
  indicate the percentage of summary you want to see, default is 10%.    
  '''
  
  if col is None:
    if attribute is not None:
      raise ValueError("With attribute column name is required. Please provide 'col' parameter")
  else:
    if col not in file.columns:
      raise ValueError('Given column is not in present.')
    
    if attribute is None:
      text = list(file[col])
      summary = ' '.join([item for item in tqdm(list(file[review])) if isinstance(item, str)] )
      return summary
    
    elif attribute not in list(file[col].unique()):
      raise ValueError('Given attribute is not present in ',col)
    
    else:
      rev = ' '.join([item for item in list(file[file[col]==attribute][review]) if isinstance(item, str)] )
      return summarize(rev,ratio)
      