# External Libraries
import torch
import streamlit as st

# Load the Model and PreTrained Weights
from model import architecture
from utils import helper

# Default Args
default_context = "The Mask of Tutankhamun is a gold mask of the 18th-dynasty ancient Egyptian Pharaoh Tutankhamun (reigned 1332–1323 BC). It was discovered by Howard Carter in 1925 in tomb KV62 in the Valley of the Kings, and is now housed in the Egyptian Museum in Cairo."
default_query = "When was the Mask of Tutankhamun discovered ?"


@st.cache(allow_output_mutation=True)
def load_model_for_app():
    # Model Definition contains default params
    BIDAF = architecture.BiDAF()
    # Pretrained Weights
    BIDAF.load_state_dict(torch.load('BIDAF.pth', map_location=torch.device('cpu')))
    BIDAF.eval()
    return BIDAF


pretrained_BIDAF = load_model_for_app()

st.title('Bidirectional Attention Flow(BiDAF) Demo')

st.header('Input')
context_text = st.text_area(label='Enter Context Here',
                            value=default_context)
query_text = st.text_input(label='Enter Query Here',
                           value=default_query)

if len(context_text) == 0 or len(query_text) == 0:
    context_text = default_context
    query_text = default_query

highlighted_context, confidence = helper.predict(context_text, query_text, pretrained_BIDAF)

st.header('Output')
st.subheader('Predicted Span ( Confidence : {0} )'.format(round(confidence, 2)))
st.markdown(highlighted_context, unsafe_allow_html=True)
