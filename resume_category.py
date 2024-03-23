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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

models_name_list = ['LogisticRegression()',
                   'KNeighborsClassifier()',
                   'DecisionTreeClassifier()',
                   'RandomForestClassifier()',
                   'SVC()',
                   'GradientBoostingClassifier()',
                   'MLPClassifier()',
                   'AdaBoostClassifier()',
                   'XGBClassifier()']

model = 'GradientBoostingClassifier()'


from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score

model_object = eval(model)
model_object.fit(conuntvectorizer_train,Y_train )
model_object.score(conuntvectorizer_test, Y_test)

pred = model_object.predict(conuntvectorizer_test[0])
print(pred)


# def fit_models(models_name: list, train_data, train_labels, test_data, test_labels) -> dict:
#     result = []
#     for i, model in enumerate(models_name):
#         try:
#             model_object = eval(model)
#             model_object.fit(train_data, train_labels)
#             print(f'{str(model)}:\n \ttraining Score: {model_object.score(train_data, train_labels)}')
#             print(f"\ttest Score: {model_object.score(test_data, test_labels)}")
#             pred = model_object.predict(test_data)
#             models_dict = {"model_name": str(model),
#                             "model": model_object,
#                             "metrics": {"names":[
#                                                 "Accuracy",
#                                                 "Sensitivity",
#                                                 "Specificity",
#                                                 "precision",
#                                                 "Recall",
#                                                 "F1-score",
#                                                 "G-Mean",
#                                                 "MCC"],
#                                        "values":[
#                                                  round(accuracy_score(test_labels, pred), 2),
#                                                  round(sensitivity_score(test_labels, pred, average="micro"), 2),
#                                                  round(specificity_score(test_labels, pred, average="micro"), 2),
#                                                  round(precision_score(test_labels, pred, average="micro"), 2),
#                                                  round(recall_score(test_labels, pred, average="micro"), 2),
#                                                  round(f1_score(test_labels, pred, average="micro"), 2),
#                                                  round(geometric_mean_score(test_labels, pred, average="micro"), 2),
#                                                  round(matthews_corrcoef(test_labels, pred), 2)]
                                       
#                                        }
#                           }
#             result.append(models_dict)
# #             print(models_dict)
# #             print(result)
#         except:
#             pass
#     return result


# results = fit_models(models_name_list, conuntvectorizer_train, Y_train, conuntvectorizer_test, Y_test)
# results