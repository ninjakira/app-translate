import pickle
import pandas as pd
import yake
from transformers import pipeline
import streamlit as st


# title
"""
# Translate English Text to German and French
"""


# user input
with st.form("my_form"):
  text_en = st.text_area("Input Text (English) to Translate:")

  # Every form must have a submit button.
  submitted = st.form_submit_button("Submit")
  
  if submitted:
    with st.spinner(text='In progress...'):

      # import model
      translate_en_de = pipeline("translation_en_to_de")
      translate_en_fr = pipeline("translation_en_to_fr")

      # translate
      text_de = translate_en_de(text_en)[0]['translation_text']
      text_fr = translate_en_fr(text_en)[0]['translation_text']



      # output (translation)
      st.subheader('Input Text (English):')
      st.write(text_en)

      st.subheader('German Text:')
      st.write(text_de)

      st.subheader('French Text:')
      st.write(text_fr)

      # output (keywords)
      extractor = yake.KeywordExtractor(lan = 'en', \
                                      n = 1, \
                                      dedupLim=0.9, \
                                      top=10, \
                                      features=None)

      key_en = pd.DataFrame(extractor.extract_keywords(text_en))[0]
      key_de = pd.DataFrame(translate_en_de(list(key_en)))
      key_fr = pd.DataFrame(translate_en_fr(list(key_en)))
      
      df = pd.concat([key_en, key_de, key_fr], axis=1)
      df.columns = ["English", "German", "French"]
      
      st.subheader('Keywords:')
      st.dataframe(df)





