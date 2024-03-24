
from joblib import Parallel, delayed 
import joblib 
import pandas as pd
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
import gensim

class AccessCategory:
    def __init__(self):

      self.loaded_model = joblib.load(r'category_clf_1.pkl') 
      self.x_train = joblib.load(r'x_train_1.pkl') 
      self.stop_words= stopwords.words('english')
      
    def preprocess(self,txt):
        # STEMMER = nltk.stem.porter.PorterStemmer()
        txt = txt.lower()
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        txt = nltk.tokenize.word_tokenize(txt)
        english_stopwords = nltk.corpus.stopwords.words('english')
        txt = [w for w in txt if not w in english_stopwords]

        return ' '.join(txt)





    def remove_stop_words (self,text):
      
      result = []
      for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in self.stop_words:
          result.append(token)

      return result


    def run(self,raw_data):
      resume = self.preprocess(raw_data)
      resume = self.remove_stop_words(resume)

      
      df = pd.read_csv(r'C:\Users\soham\Desktop\Spunky\hacka\Resume\Resume.csv')

      
      df.drop(columns = ['ID', 'Resume_html'], inplace = True)
      df['Resume'] = df['Resume_str'].apply(lambda w: self.preprocess(w))
      df.pop('Resume_str')

      categories = np.sort(df['Category'].unique())
      df_categories = [df[df['Category'] == category].loc[:, ['Resume', 'Category']] for category in categories]
      from sklearn.preprocessing import LabelEncoder

      label_encoder = LabelEncoder()
      label_encoder.fit_transform(df['Category'])
      print(label_encoder.classes_)


      from sklearn.feature_extraction.text import CountVectorizer

      cv1 = CountVectorizer()
      cv1.fit_transform(self.x_train)

      document = resume


      resume_vectors = cv1.transform(document)



      categories = ['ACCOUNTANT' 'ADVOCATE' 'AGRICULTURE' 'APPAREL' 'ARTS' 'AUTOMOBILE'
      'AVIATION' 'BANKING' 'BPO' 'BUSINESS-DEVELOPMENT' 'CHEF' 'CONSTRUCTION'
      'CONSULTANT' 'DESIGNER' 'DIGITAL-MEDIA' 'ENGINEERING' 'FINANCE' 'FITNESS'
      'HEALTHCARE' 'HR' 'INFORMATION-TECHNOLOGY' 'PUBLIC-RELATIONS' 'SALES'
      'TEACHER']

      idx = self.loaded_model.predict(resume_vectors)
      idx = label_encoder.inverse_transform(idx)
      
      return idx





if __name__ == "__main__":
    p = AccessCategory()
    res = p.run('''         DIRECTOR OF NATIONAL SALES- US. HEALTHCARE           Executive Profile     SALES AND BUSINESS DEVELOPMENT EXECUTIVE Successful in sales management and business development at the local, regional, and national levels. Hands-on manager with highly developed negotiation skills. Provide sound budgeting, financial, and forecasting management. Creative problem solver who drives revenue, resolves conflict, and consistently exceeds sales goals.       Skill Highlights          Leadership/communication skills  Business operations organization  Client account management  Budgeting expertise  Negotiations expert  Employee relations  Self-motivated  Market research and analysis  Customer-oriented  Microsoft Family Products   Customer CRM       GPO and IDN targeting   Vendor and Distributor Relations  National Business Development  Regional Business Development  Local Business Development  Forecasting  C-Suite Executive Targeting   Exceed Profit and Sales Goals   Problem Solver   Sales Management             Core Accomplishments     45% Healthcare division growth in 2014   500% growth of Healthcare active business pipeline   Developed, managed, supported sales budget that exceeded 20 million dollars   Exceeded sales and profit goals by 40% plus in 2010, 2011, 2012, 2013, 2014  Grew Northeast Region into largest and most profitable territory in company 2012-2014  Largest territory margin increase in company 2012-2014  Took territory from 5 % under contract to 65% (highest % in company) 2012-2014  Highest new account margin in company 2013-2014  Multi-Year contest winner        Professional Experience      Director of National Sales- US. Healthcare     March 2014   to   Current     Company Name   ï¼   City  ,   State      Responsible for  leading and overseeing all national sales functions for healthcare segment consisting of  medical gases, maintenance/certification services, and durable medical equipment   Develop strategies to improve customer experience while increasing sales margins within hospital, dental clinics, skilled nursing centers, medical equipment and healthcare services segments.   Manage divisional budgets/P&L, forecasting, sales, supply chain management, strategic direction and business planning for national sales representatives and supply chain engineers   Identify key strategic relationships with suppliers in medical equipment, medical gas supplies, maintenance and certification services, GPO and buying groups to increase margin and sales  Created new healthcare sales verticals and channel sales opportunities   Manage and develop regional, national, and local distributor relationships for healthcare segment  Responsible for client related risk assessment, action planning, project development, and  implementation   Project manager of all new healthcare facility construction opportunities  Developed all healthcare training and marketing material for internal and external personnel   Prospect, assess, mentor, and develop all fortune 500 healthcare opportunities in Nashville and with top tier US national customers  Train national sales team in all aspects of healthcare related sales material including proposals, product offerings, and consultative healthcare sales tactics   Support day to day sales activities for all reps   Develop reporting capabilities for customer dashboards and key performance indicators for healthcare division  Developed systems, policies, and procedures for internal customer service and data entry staff.   Present all major proposals to clients, negotiate pricing, review contracts, and define service expectations           National Accounts Manager- Northeast Region     June 2012   to   March 2014     Company Name   ï¼   City  ,   State      Industries serviced include hospitals, skilled nursing facilities, clinics, retail sporting goods, and industrial wholesale contractor outlets for medical/industrial/retail gases and equipment   Responsible for overseeing all business development activity in northeast territory that included all customer activities, customer service, budgeting, forecasting, contract negotiation, and billing.    Attained new business via campaign management,  direct selling, prospect qualification, value capture analysis through consultative selling techniques   Coordinated all internal company activities with external partners to deliver solutions to clients   Managed and maintained relationships with key national and regional distributors   Achieved highest customer service ranking within company   Managed, developed, and maintained highest profit and sales territory for entire company that included top 2 industrial accounts, #1 retail account, and #1 hospital account.   Maintained highest activity levels within company for meetings, proposals, and new business sold.           Business Development Manager     June 2006   to   April 2012     Company Name   ï¼   City  ,   State      Responsible for managing all aspects of engineering business development and sales for Delaware and New Jersey to medical device, pharmaceutical, industrial manufacturing, electronic manufacturing, and R&D organizations. (DuPont, Dentsply International, Siemens, W.L. Gore, Goodrich, Chrysler, General Motors, T.A. Instruments, FMC BioPolymer)   Exceed weekly actively goals with 15 + meetings, 3 client lunches, 100 + daily cold calls, 100 self-generated leads  Responsible for customer analysis, developing sourcing strategies, identifying screening requirements per customer, coordinating selection and compliance processes, identifying K.P.I. and initiating formal procedures for follow-up and client saturation /satisfaction   Coordinate and manage all internal responsibilities for various internal departments   Identify and build relationships with all key decision makers and influencers that include: Direct and Indirect Hiring Managers,  Provide a consultative and results driven process to clients that is accompanied by continuous follow-up           Education      B.A   :   Marketing  ,   2006    Bloomsburg University   ï¼   City  ,   State              Professional Training      Karrass Effective Negotiating Seminar   Linde Pro Sales Training  Sales Performance International-Solution Sales   Sales Performance Internal-Management Training   Challenger Sales Training  Completed Advanced Sales Training I  Consultative Sales Training Situational Leadership I  Behavioral Interviewing Training  Advanced Lead Generation Techniques and Diversity Training     ''')

    print(res)