############################################
# Import Modules
############################################
import streamlit as st
import json
import pickle
import pandas as pd
import re
import plotly.express as px


############################################
# Config and Constants
############################################
st.set_page_config(page_title="Ontology | Social Media Analytics", page_icon=None,
                   layout="centered", initial_sidebar_state="auto", menu_items=None)

############################################
# Functions Definition
############################################


@st.cache_data
def load_ontology():
    return json.load(open("backend/ontology.json", "rb"))


def filter_ontology(ontology):
    results = {}

    for industry in ontology.keys():
        results[industry] = {}
        for aspect in ontology[industry].keys():
            count = 0
            for adjective in ontology[industry][aspect].keys():
                count += len(ontology[industry][aspect][adjective])

            results[industry][aspect] = count

    return results


############################################
# Main
############################################
ontology = load_ontology()
filtered_ontology = filter_ontology(ontology)


st.title("Ontology Design")
st.write("This is a demonstration of some of the concepts used during the design of the ontology. The entire ontology is not shown here due to size reasons, but the selected and filtered ontology can already demonstrate how it was used to perform various tasks.")

st.header('Overview', divider='gray')
st.write("Below shows a glimpse of the filtered ontology, where each industry has 3-4 aspects, and are further broken down into more subtopics downstream. The number indicates the number of keywords or phrases (unigrams, bigrams, and trigrams) extracted from the scraped dataset that represents this subtopic.")
st.write(filtered_ontology)


################################################################
# Design start
#################################################################

st.header('Designing the Ontology', divider='gray')
st.write("The ontology was designed using three main stages, generating N-grams from text reviews, using Zipf's Law for importance estimation, and similarity scores for further filtering.")

################################################################
# N GRAMS
################################################################

st.subheader('N-Grams from Text Reviews', divider='gray')
st.write("N grams are used to take out the most frequency terms used in each industry for us to pinpoint which keywords to pick that represents the different aspects within the industry.")

selected_industry = st.selectbox('Select Industry', options=[
                                 'Catering', 'Banking', 'Hotel', 'Attraction'], key="selected_industry")

ngrams = json.load(open('backend/ngrams.json', 'rb')
                   ).get(selected_industry.lower())


st.subheader('Unigram')
unigrams = ngrams['unigram']
filtered_ngrams = {k: v for k, v in dict(sorted(unigrams.items(
), key=lambda x: x[1], reverse=True)).items() if not re.fullmatch(r"\d+", k)}

plotted_grams = {k: v for k, v in filtered_ngrams.items()
                 if k in list(filtered_ngrams.keys())[:10]}

st.write(px.bar(pd.DataFrame(
    {'unigrams': plotted_grams.keys(), 'frequency': plotted_grams.values()}), x='frequency', y='unigrams', orientation='h'))

with st.expander("See All N-Grams"):
    st.write(filtered_ngrams)


# st.subheader('Bigram')
# bigrams = ngrams['bigram']
# filtered_ngrams = {k: v for k, v in dict(sorted(bigrams.items(
# ), key=lambda x: x[1], reverse=True)).items() if not re.fullmatch(r"\d+", k)}
# plotted_grams = {k: v for k, v in filtered_ngrams.items()
#                  if k in list(filtered_ngrams.keys())[:10]}

# st.write(px.bar(pd.DataFrame(
#     {'bigrams': plotted_grams.keys(), 'frequency': plotted_grams.values()}), x='frequency', y='bigrams', orientation='h'))

# with st.expander("See All N-Grams"):
#     st.write(filtered_ngrams)


# st.subheader('Trigram')
# trigrams = ngrams['trigram']
# filtered_ngrams = {k: v for k, v in dict(sorted(trigrams.items(
# ), key=lambda x: x[1], reverse=True)).items() if not re.fullmatch(r"\d+", k)}
# plotted_grams = {k: v for k, v in filtered_ngrams.items()
#                  if k in list(filtered_ngrams.keys())[:10]}

# st.write(px.bar(pd.DataFrame(
#     {'trigrams': plotted_grams.keys(), 'frequency': plotted_grams.values()}), x='frequency', y='trigrams', orientation='h'))

# with st.expander("See All N-Grams"):
#     st.write(filtered_ngrams)
