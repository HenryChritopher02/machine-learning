import streamlit as st
import pandas as pd
import numpy as np

st.title('🎈 Machine learning App')

st.write('Hello world!')

df = pd.read_csv(r'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/bace1_standardized.csv',index=False)
df
