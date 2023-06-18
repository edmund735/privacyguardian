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

if st.button('Accept'):
    st.write("Terms of agreement accepted.")
elif st.button('Decline'):
    st.write("Terms of agreement declined.")
elif st.button('Accept and leave a statement of objection'):
    st.write("What data do you not want to have collected?")
    bad_data = st.text_input('Data objections:')
    bad_ret = st.text_input('Retention objections:')
    
