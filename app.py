import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

#Sidebar
with st.sidebar:
    st.title('Chat Application 💭')
    st.markdown('''
    # About
    This app is LLM (Large Language Model) powered chatbot built using:
                
    -[Streamlit](https://streamlit.io/)  
    -[LangChain](https://www.langchain.com/)     
    -[OpenAI](https://platform.openai.com/docs/models)                                        
    ''')
    add_vertical_space(5)
    st.write('Made by Yukta Kumawat😎')

load_dotenv()

def main():
    st.header("Chat with PDF 🖇️")

    
    #Upload PDF
    pdf=st.file_uploader("upload your PDF", type='pdf')
    

    if pdf is not None:
        pdf_reader= PdfReader(pdf)

        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

        chunks= text_splitter.split_text(text=text)

    #embeddings
    try:
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore=pickle.load(f)
            #st.write('Embeddings loaded from the disk')  
        else:
            embeddings= OpenAIEmbeddings() 
            VectorStore= FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        #Accept user questions
        query= st.text_input("Ask questions about your PDF file:")        
        #st.write(query)
        
        if query:
            docs=VectorStore.similarity_search(query=query, k=3)
            
            llm= OpenAI(model_name='gpt-3.5-turbo')
            chain= load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    except:
        st.write("Please select a pdf file to continue")

        
if __name__=='__main__':
    main()    