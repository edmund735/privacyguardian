"""
app
"""

import streamlit as st

st.title('PrivacyGuardian UI')

pol = st.text_input('Privacy policy:')

# send prompt to chatgpt to figure out what data it collects
data_coll = ''
ret_pol = ''
intr_score = ''

st.write("Types of data this company collects: " + data_coll)
st.write("Retention policy: " + ret_pol)
st.write("Intrusiveness score: " + intr_score)

st.button('Accept')
st.button('Accept and leave a statement of objection')
# implement stuff for autogeneration of statement of obj
st.button('Decline')