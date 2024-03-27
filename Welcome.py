import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

# Define the path to the 'uploaded_files' directory
msaproject_path = 'uploaded_files'

# Function to save uploaded file within the 'msaproject' folder structure
def save_uploaded_file(uploadedfile):
    upload_folder = os.path.join(msaproject_path, 'uploaded_files')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path  # Return the path where the file was saved

st.set_page_config(page_title="Segmentation and Sentiment Analysis", layout="wide")

# Page header
st.title("Segmentation and Sentiment Analysis for Women's E-commerce Clothing")

# Create two columns for project details and file uploader
col1, col2 = st.columns(2)

# Use the first column for Project Details
with col1:
    st.markdown("""
        **Objective:** The primary aim of this project is to enhance the understanding of customer preferences and market segments in women's e-commerce clothing. By analyzing data from online sources, we intend to identify patterns and sentiments associated women's e-commerce clothing, thereby facilitating a more customer-centric approach to marketing and product development.

        **Proposed Solution:** We propose to create a comprehensive analytics dashboard that will ingest raw e-commerce data, apply data cleaning and preprocessing techniques, and utilize clustering algorithms to segment the market based on customer behaviors and preferences. The sentiment analysis component will evaluate customer reviews and feedback to gauge satisfaction levels and to extract actionable insights. This multifaceted approach will empower stakeholders to make data-driven decisions and tailor their offerings to better meet the needs of distinct customer groups.
    """)

# Use the second column for Data File Uploader
with col2:
    st.subheader("Data File Uploader")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader")
    submit_button_pressed = False  # Button state tracking
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File saved to: {file_path}")
        submit_button_pressed = st.button("Submit", key="submit_btn")

# Check if the submit button was pressed and an uploaded file is present
if submit_button_pressed:
    # Navigate to the next page (page2)
    switch_page("page2_UncleanData")

# Footer
st.text("Developed by Rahul Jiandani, Shreyash Nadgouda & Vanshika Nijhawan")
