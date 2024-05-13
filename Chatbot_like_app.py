import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np


from agents import get_llm
from agents import smartdataframe_respone,response_agent,code_agent,graph_agent



model_name = ['llama3-70b-8192','mixtral-8x7b-32768','llama3-8b-819','gemma-7b-it']



def show_image():
    buf = io.BytesIO()   
    plt.savefig(buf, format="png")  
    buf.seek(0)
    image_buffers.append(buf)
    st.image(buf, use_column_width=True)






st.title("Data Analysis with AI Agents")
# st.markdown("<h4 style='text-align: right;'>Owner -- Raj Singh Yadav</h1>", unsafe_allow_html=True)
st.header(' ', divider='rainbow')
uploader_file = st.sidebar.file_uploader("Upload a CSV file", type= ["csv"])






if uploader_file is not None:
    data = pd.read_csv(uploader_file)

    if st.sidebar.checkbox('Show dataframe'):
        # st.write(data.head(5))
        add_slider = st.sidebar.slider(
        'Number of rows u want to See: ',
         0, 10,5
        )

        st.markdown("<h5 style='text-align: center;'>DataFrame </h1>", unsafe_allow_html=True)
        st.write(f"First {add_slider} columns of DataFrame ")
        st.write(data.head(add_slider))

        st.write(f"{add_slider} Sample from the DataFrame ")
        st.write(data.sample(n=add_slider))

        st.write(f"Describe the DataFrame ")
        st.write(data.describe())
        



    if st.sidebar.checkbox("Show Data info"):
        st.markdown("<h5 style='text-align: center;'>DataFrame INFO </h1>", unsafe_allow_html=True)
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


    model = st.sidebar.selectbox(
    'Which model u want to chose ?',
    model_name)
    st.sidebar.text(f'Selected model : {model}')
    

    llm = get_llm(model)



    st.markdown("<h4 style='text-align: center;'>Query your DataFrame </h1>", unsafe_allow_html=True)


            # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    # React to user input
    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = f"Echo: {prompt}"
    


        # with st.spinner("Generating response..."):
        # Create a list to store the image buffers
        image_buffers = []

        st.markdown("<h6 style='text-align: center;'>SmartDataFrame Output</h1>", unsafe_allow_html=True)
        # st.write(smartdataframe_respone(data,prompt,llm))
        response = smartdataframe_respone(data,prompt,llm)
        with st.chat_message("Assistant 1"):
            st.markdown(response)
            # show_image()
            
        



        st.markdown("<h6 style='text-align: Center;'>Agent Output</h1>", unsafe_allow_html=True)
        # st.write(response_agent(data,prompt,llm))
        response = response_agent(data,prompt,llm)
        with st.chat_message("Assistant 2"):
            st.markdown(response)
            # show_image()
        


        st.markdown("<h6 style='text-align: center;'>Code Agent Output</h1>", unsafe_allow_html=True)
        # st.write(code_agent(data,prompt,llm))
        response = code_agent(data,prompt,llm)
        with st.chat_message("Assistant 3"):
            st.markdown(response)
            # show_image()



        st.markdown("<h6 style='text-align: center;'>Graph Agent Output</h1>", unsafe_allow_html=True)
        # st.write(graph_agent(data,prompt,llm)) 
        response = graph_agent(data,prompt,llm)
        with st.chat_message("Assistant 4"):
            st.markdown(response)
            show_image()


        st.session_state.messages.append({"role": "assistant", "content": response})










