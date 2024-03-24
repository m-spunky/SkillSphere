import pandas as pd
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import stopwords
import gensim
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\Users\soham\Desktop\Spunky\hacka\Resume\Resume.csv')

df.drop(columns = ['ID', 'Resume_html'], inplace = True)

STEMMER = nltk.stem.porter.PorterStemmer()


def preprocess(txt):
    txt = txt.lower()
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = nltk.tokenize.word_tokenize(txt)
    english_stopwords = nltk.corpus.stopwords.words('english')
    txt = [w for w in txt if not w in english_stopwords]

    return ' '.join(txt)


df['Resume'] = df['Resume_str'].apply(lambda w: preprocess(w))

df.pop('Resume_str')


categories = np.sort(df['Category'].unique())
print(categories)
df_categories = [df[df['Category'] == category].loc[:, ['Resume', 'Category']] for category in categories]


from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def remove_stop_words (text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
      result.append(token)

  return result

df['clean'] = df['Resume'].apply(remove_stop_words).astype(str)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['numerical_labels'] = label_encoder.fit_transform(df['Category'])



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df['clean'], df['numerical_labels'], test_size = 0.2, random_state=42)


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer()
conuntvectorizer_train = vectorizer.fit_transform(X_train).astype(float)

conuntvectorizer_test = vectorizer.transform(X_test).astype(float)


# Models
from sklearn.ensemble import GradientBoostingClassifier

model = 'GradientBoostingClassifier()'


model_object = eval(model)
model_object.fit(conuntvectorizer_train,Y_train )
model_object.score(conuntvectorizer_test, Y_test)



from joblib import Parallel, delayed 
import joblib 


joblib.dump(X_train,'x_train_1.pkl') 
joblib.dump(model_object,'category_clf_1.pkl') 



