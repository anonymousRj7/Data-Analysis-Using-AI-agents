import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np


from agents import get_llm
from agents import smartdataframe_respone,response_agent,code_agent,graph_agent



model_name = ['llama3-70b-8192','mixtral-8x7b-32768','llama3-8b-819','gemma-7b-it']








st.title("Data Analysis with AI Agents")
# st.markdown("<h4 style='text-align: right;'>Owner -- Raj Singh Yadav</h1>", unsafe_allow_html=True)
st.header(' ', divider='rainbow')
uploader_file = st.sidebar.file_uploader("Upload a CSV file", type= ["csv"])


# Create a list to store the image buffers
image_buffers = []


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
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):


                st.markdown("<h4 style='text-align: center;'>SmartDataFrame Output</h1>", unsafe_allow_html=True)
                st.write(smartdataframe_respone(data,prompt,llm))
                buf = io.BytesIO()   
                plt.savefig(buf, format="png")  
                buf.seek(0)
                image_buffers.append(buf)

                st.markdown("<h4 style='text-align: Center;'>Agent Output</h1>", unsafe_allow_html=True)
                st.write(response_agent(data,prompt,llm)) 
                buf = io.BytesIO()   
                plt.savefig(buf, format="png")  
                buf.seek(0)
                image_buffers.append(buf)

                st.markdown("<h4 style='text-align: center;'>Code Agent Output</h1>", unsafe_allow_html=True)
                st.write(code_agent(data,prompt,llm))
                buf = io.BytesIO()   
                plt.savefig(buf, format="png")  
                buf.seek(0)
                image_buffers.append(buf)   


                st.markdown("<h4 style='text-align: center;'>Graph Agent Output</h1>", unsafe_allow_html=True)
                st.write(graph_agent(data,prompt,llm)) 
                buf = io.BytesIO()   
                plt.savefig(buf, format="png")  
                buf.seek(0)
                image_buffers.append(buf)
                




        else:
            st.warning("Please enter a prompt!")





# Create columns to display the images
num_columns = len(image_buffers)
if num_columns !=0 :
    columns = st.columns(num_columns)

# Display each image in a separate column
for i, buf in enumerate(image_buffers):
    with columns[i]:
        st.markdown(f"**Agent {i+1}**")
        st.image(buf, use_column_width=True)
