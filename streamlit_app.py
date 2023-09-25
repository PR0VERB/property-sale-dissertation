import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import mean_absolute_error
from explainerdashboard import RegressionExplainer, ExplainerDashboard


df_ = pd.read_csv('melb_data.csv')
df = df_.copy()
# df = df[df.Rooms >=5]
# df = df[df.duplicated(subset=['Address'], keep=False)]

# df = df.drop_duplicates(subset = 'Address')
df['Date'] = pd.to_datetime(df['Date'])
df['Date']=df['Date'].astype(str).str.replace('-','')
df['Date'] = df['Date'].apply(lambda x: x[:-2])
df['Date'] = df.Date.astype(int)
df = df.sort_values(by = 'Date')
# df = df[df['Type'] == 'h']
# df = df.drop(['Type'],axis=1)

# remove oulier building area

Price_25 = df['Price'].quantile(0.25)
Price_75 = df['Price'].quantile(0.75)
iqr = Price_75 - Price_25
Price_lower = Price_25 - 2.5*iqr
Price_upper = Price_75 + 2.5*iqr
df = df[(df['Price'] > Price_lower) & (df['Price'] < Price_upper)]

df = df.drop(['Bedroom2','Address','CouncilArea','Propertycount', 'Suburb','Method','SellerG', 'Postcode'], axis = 1)
# kept = 'Bedroom2'
df = df.dropna()

df = pd.get_dummies(df, columns = ['Regionname','Type'], drop_first=True)
df = df.drop(df.filter(regex = 'Victoria|Eastern Metropolitan|Western Metropolitan|Northern Metropolitan|Type_t').columns, axis = 1)

x_train, x_test, y_train, y_test = tts(df.drop(['Price'],axis = 1), df['Price'], test_size = 0.15, stratify = df[['Date']], random_state = 97)

RANDOM_STATE, MAX_DEPTH, MAX_LEAF_NODES, MIN_SAMPLES_SPLIT = 97, None, None, 20
dt = DecisionTreeRegressor(max_depth = MAX_DEPTH, random_state= RANDOM_STATE, max_leaf_nodes =MAX_LEAF_NODES, min_samples_split = MIN_SAMPLES_SPLIT) # random_state = 110, no fft or cwt 
dt.fit(x_train, y_train)
y_pred  = dt.predict(x_test)
comp_dt = pd.DataFrame(data = [y_pred.round().astype(int), y_test.round().astype(int)])
comp_dt = comp_dt.transpose()
comp_dt.columns = ['Predicted', 'Actual']

mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
comp_dt['% Difference'] = (comp_dt['Predicted'] - comp_dt['Actual'])/comp_dt['Actual']
comp_dt['% Difference'].abs().mean()

explainer = RegressionExplainer(
                dt, x_test, y_test,
                # optional:
#                 cats=['Sex', 'Deck', 'Embarked'],
#                 labels=['Not survived', 'Survived']
)

db = ExplainerDashboard(explainer, title="Property Sale Price Prediction & Explainability (AUD)",
                    whatif=True, # you can switch off tabs with bools
                    shap_interaction=True,
                    decision_trees=True)
# db.run(port=8051)
db.run()
