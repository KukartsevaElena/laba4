import pandas as pd
from sklearn.metrics import f1_score
from clearml import Task, Logger
task = Task.init(project_name="lab4", task_name="my task")
logger = task.get_logger()   

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

parameters = {
    'max_depth' : 3,
    'random_state' : 10
}

parameters = task.connect(parameters)

Train = pd.read_csv('/home/vmac/laba4/dataset/train.csv')

stemmer = PorterStemmer()
sw = stopwords.words('english')

def text_preprocessing(text):
  text = text.lower() #нижний регистр
  text = re.sub(r"[^a-zA-z?.!,]+", " ", text)
  text = re.sub(r"http\S+", "", text) #Удаляем URL
  html = re.compile(r'<.*?>')
  text = html.sub(r'', text) #Удаление тегов html
  for p in string.punctuation:
    if p in text:
        text = text.replace(p, '') #Удаляю пунктуацию
  text = [stemmer.stem(word) for word in text.split() if word not in sw] #стемминг и удаление стоп-слов
  text = " ".join(text)
  return text

Train['text'] = Train['text'].apply(lambda x: text_preprocessing(x))
X_train = Train['text']
Y_train = Train['target']

tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_vectorizer.fit(X_train)
X_train, Y_train = tfidf_vectorizer.transform(Train['text']),Train['target']

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth = parameters['max_depth'], 
                              random_state = parameters['random_state'])
classifier_tfidf = tree.fit(X_train,Y_train)
pred = classifier_tfidf.predict(X_train)
f1_train = f1_score(Y_train,pred)
f1_train

Logger.current_logger().report_scalar(title='f1_train', series='f1_score', value=f1_train, iteration=1)
task.upload_artifact(name='f1_train', artifact_object={'f1_score':f1_train})