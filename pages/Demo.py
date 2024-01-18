############################################
# Import Modules
############################################
# import guidance
import streamlit as st
import pandas as pd
import ast
import pickle
from scipy.sparse import hstack
import torch
import torch.nn.functional as F
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import json
import os
# import openai
# from dotenv import load_dotenv
import numpy as np
from collections import Counter, defaultdict


############################################
# Config and Constants
############################################

st.set_page_config(page_title="Demo | Social Media Analytics", page_icon=None,
                   layout="centered", initial_sidebar_state="auto", menu_items=None)

if 'd_data' not in st.session_state:
    st.session_state['d_data'] = None

if 'd_sample' not in st.session_state:
    st.session_state['d_sample'] = ""

if 'd_ont' not in st.session_state:
    st.session_state['d_ont'] = None

if 'd_topic' not in st.session_state:
    st.session_state['d_topic'] = None

if 'd_sentiment' not in st.session_state:
    st.session_state['d_sentiment'] = None

if 'd_sentiment_score' not in st.session_state:
    st.session_state['d_sentiment_score'] = None

if 'd_explain' not in st.session_state:
    st.session_state['d_explain'] = None

if 'd_demo_count' not in st.session_state:
    st.session_state['d_demo_count'] = -1

# program = guidance("""
# Review: {{review_text}}
# Given the above review and its sentiment score of {{sentiment_score}}, we aim to provide a comprehensive explanation that delves into various aspects of '{{topic_classification}}'. We take into account specific attributes and keywords identified from the review to shed light on the factors contributing to the overall sentiment.

# For each identified aspect, we will explore how the associated attributes and keywords play a role in shaping the {{sentiment_label}} sentiment:
# By examining these aspects and their corresponding attributes, we aim to provide a clear and structured explanation, highlighting how they collectively contribute to the positive sentiment expressed in the review.

# {{#each path_list}}
# - Analyzing '{{this.path}}' (mentioned {{this.count}} times):
#   Keywords: {{this.keywords}}
#   {{~gen "explaination" temperature=0 max_tokens=100 list_append=True}}
# {{~/each}}

# """)


############################################
# Functions Definition
############################################

def get_data():
    data_path = 'backend/demo_data.csv' if DEMO_MODE else 'backend/all_data.csv'
    st.session_state['d_data'] = pd.read_csv(data_path)
    return st.session_state['d_data']


@st.cache_resource
def get_topic_model():
    return pickle.load(open('backend/topic_model_tfidf_svc_ontology.sav', 'rb'))


@st.cache_resource
def get_vectorizer():
    return pickle.load(open('backend/vectorizer.pickle', 'rb'))


@st.cache_resource
def get_bert_model_and_tokenizer():
    # Load pre-trained model and tokenizer
    # You can use other BERT variants as well
    model_name = 'distilbert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained('backend')

    return model, tokenizer


def set_sample():
    if DEMO_MODE:
        st.session_state['d_demo_count'] += 1
        st.session_state['d_demo_count'] = st.session_state['d_demo_count']%3
        st.session_state['d_sample'] = st.session_state['d_data']['content'].iloc[st.session_state['d_demo_count']]
    else:
        st.session_state['d_sample'] = st.session_state['d_data'][st.session_state["d_data"]["explaination"].notnull()].sample(1)[
        'content'].iloc[0]
    st.session_state['d_ont'] = None
    st.session_state['d_topic'] = None
    st.session_state['d_sentiment'] = None
    st.session_state['d_sentiment_score'] = None
    st.session_state['d_explain'] = None


def set_ont():
    st.session_state['d_ont'] = df[df['content'] == st.session_state['d_sample']][[
        d for d in df.columns if '|' in d]]


def set_topic():
    random_review = df[df['content'] == st.session_state['d_sample']]
    text = random_review['content'].iloc[0]
    binaries = random_review[random_review.columns[6:19]
                             ].to_numpy().astype(float)

    text_vectorized = vectorizer.transform([text])
    input_features = hstack([text_vectorized, binaries])

    st.session_state['d_topic'] = topic_model.predict(input_features)[0]


def set_sentiment():
    # Tokenize and encode the sentences
    encoded_inputs = bert_tokenizer(
        [st.session_state['d_sample']], padding=True, truncation=True, return_tensors='pt')

    input_ids = tf.convert_to_tensor(encoded_inputs['input_ids'])
    attention_mask = tf.convert_to_tensor(encoded_inputs['attention_mask'])

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=1)
    sentiment_score = round(tf.nn.softmax(
        logits, axis=1).numpy()[0][1]*5, 0)
    predicted_label = tf.math.argmax(probabilities, axis=1).numpy()[0]

    st.session_state['d_sentiment'] = 'positive' if predicted_label == 1 else 'negative'
    st.session_state['d_sentiment_score'] = sentiment_score

    # st.session_state['d_sentiment'] = df[df['content'] == st.session_state['d_sample']]['sentiment_score'].iloc[0]


def set_explanation():
    try:
        st.session_state['d_explain'] = ast.literal_eval(
            df[df['content'] == st.session_state['d_sample']]['explaination'].iloc[0])
    except Exception:
        set_sample()


############################################
# Main
############################################
st.title("End to End Demo")
st.write("This part demonstrates the end to end flow of the algorithm developed with examples that were generated during the study.")

DEMO_MODE = st.toggle('Demo Mode', value=True, label_visibility="visible")

df = get_data()
st.session_state['d_data'] = df

ontology = json.load(open('backend/ontology.json', 'r'))
topic_model = get_topic_model()
vectorizer = get_vectorizer()
bert_model, bert_tokenizer = get_bert_model_and_tokenizer()

# load_openai()

############################################
# Sample Review
############################################
st.header('Sample a Random Review', divider='gray')
st.write("In order to limit the size of the app, we have sampled 400~ text reviews from the dataset for demonstration. To begin, click the button below.")
st.button('Randomly Select 1 Text Review',
          on_click=set_sample, use_container_width=True)
if st.session_state['d_sample'] != "":
    st.subheader("Random Review")
    with st.chat_message("human"):
        st.write(st.session_state['d_sample'])

############################################
# Map Ontology
############################################
# if st.session_state['d_sample'] != "":
#     st.header('Map the Ontological Features', divider='gray')
#     st.write("Based on the ontology designed, map out what keywords were matched with binary features.")
#     st.button('Map on Ontology Features', on_click=set_ont, use_container_width=True)

#     if st.session_state['d_ont'] is not None:
#         ont_features = st.session_state['d_ont'].iloc[0].tolist()

#         st.subheader("Catering", divider='gray')
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric(label="Environment", value=ont_features[0])
#         with col2:
#             st.metric(label="Price", value=ont_features[1])
#         with col3:
#             st.metric(label="Cuisine", value=ont_features[2])
#         with col4:
#             st.metric(label="Service", value=ont_features[3])

#         st.subheader("Hotel", divider='gray')
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric(label="Environment", value=ont_features[4])
#         with col2:
#             st.metric(label="Price", value=ont_features[5])
#         with col3:
#             st.metric(label="Service", value=ont_features[6])

#         st.subheader("Attraction", divider='gray')
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric(label="Environment", value=ont_features[7])
#         with col2:
#             st.metric(label="Price", value=ont_features[8])
#         with col3:
#             st.metric(label="Transportation", value=ont_features[9])

#         st.subheader("Banking/Finance/Insurance", divider='gray')
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric(label="Product", value=ont_features[10])
#         with col2:
#             st.metric(label="Service", value=ont_features[11])
#         with col3:
#             st.metric(label="Price", value=ont_features[12])

############################################
# Topic Classification
############################################
if st.session_state['d_sample'] != "":
    st.header('Topic Classification', divider='gray')
    st.write("Perform TDIDF as well as include the ontological binary features into the SVM topic classification model.")

    st.button('Perform Topic Classification',
              on_click=set_topic, use_container_width=True)

    if st.session_state['d_topic'] is not None:
        with st.chat_message("human", avatar="🤖"):
            st.write("The text is classified to be:")
            st.text(st.session_state['d_topic'])

        # st.write(get_masked_subcat_and_keywords(
        #     st.session_state['d_topic'], st.session_state['d_sample']))
############################################
# Sentiment Analysis
############################################
if st.session_state['d_sample'] != "" and st.session_state['d_topic'] is not None:
    st.header('Sentiment Analysis', divider='gray')
    st.write(
        "Perform sentiment analysis using a BERT model for positive or negative sentiment.")

    st.button('Perform Sentiment Analysis',
              on_click=set_sentiment, use_container_width=True)

    if st.session_state['d_sentiment'] is not None:
        with st.chat_message("human", avatar="🤖"):
            st.write("The sentiment using finetuned BERT model is:")
            st.text(
                f"{st.session_state['d_sentiment']} ({st.session_state['d_sentiment_score']})")

############################################
# Explainer
############################################
if st.session_state['d_sample'] != "" and st.session_state['d_topic'] is not None and st.session_state['d_sentiment'] is not None:
    st.header('Explainer', divider='gray')
    st.write("Based on the ontology and keywords extraction, provide an explanation on the sentiment score,")

    st.button('Provide Explanation', on_click=set_explanation,
              use_container_width=True)

    if st.session_state['d_explain'] is not None:
        st.write(st.session_state['d_explain'])

        st.button('Restart', on_click=set_sample, use_container_width=True)
