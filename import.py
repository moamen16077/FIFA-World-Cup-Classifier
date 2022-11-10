import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.write("""
# Levels Company 
### For Marketing strategies
The 2022 ***FIFA World Cup*** is scheduled to be the 22nd running of the FIFA World Cup competition, the quadrennial 
international men's football championship contested by the senior national teams of the member 
associations of FIFA. It is scheduled to take place in **Qatar** from **20 November to 18 December 2022**.
""")

st.sidebar.header('See who will win')
df_org = pd.read_csv("worldCup_dataset.csv" , sep = "," , encoding = 'utf-8')
df_clean = pd.read_csv("fifa world cup clean.csv" , sep = "," , encoding = 'utf-8')

def user_input_features():
    Team_1 = st.sidebar.selectbox('Team 1', (df_org['home_team'].unique()))
    Team_2 = st.sidebar.selectbox('Team 2', (df_org['away_team'].unique()))
    Nutral = st.sidebar.selectbox('Neutral', (df_org['neutral'].unique()))
    Country = st.sidebar.selectbox('Country will play on it', (df_org['country'].unique()))
    data = {'Team_1': Team_1,
            'Team_2': Team_2,
            'Nutral': Nutral,
            'Country': Country}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('previous Matches')
st.write(df_org.tail(10)) 


x = df_clean.drop(['target'],axis=1).values
y = df_clean['target'].values

dt_clf = DecisionTreeClassifier()
st.write("""
# Prediction 
### Team 1
""")
