import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
import os

# Function to plot histograms
def plot_histograms(data, column):
    fig = px.histogram(data, x=column, title=f"Histogram of {column}",
                       color_discrete_sequence=['mediumslateblue'],
                       barmode='overlay',
                       barnorm='percent',
                       opacity=1,
                       nbins=20)
    fig.update_layout(bargap=0.1)
    return fig

# Function to plot pie chart
def plot_pie_chart(data, column):
    fig = px.pie(data, names=column, title=f"Pie Chart of {column}",
                 hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# Function to plot boxplots
def plot_boxplot(data, column):
    fig = px.box(data, y=column, title=f"Boxplot of {column}",
                 color_discrete_sequence=['mediumslateblue'])
    return fig

# Function to display data summary
def display_data_summary(data):
    st.write(f"### Count before dropping NA: {len(data)}")
    
    summary_data = {
        'Column': [],
        'Data Type': [],
        'Unique Values': [],
        'Missing Values': [],
        'Mean (Numeric Columns)': []
    }
    for col in data.columns:
        summary_data['Column'].append(col)
        summary_data['Data Type'].append(data[col].dtype)
        summary_data['Unique Values'].append(data[col].nunique())
        summary_data['Missing Values'].append(data[col].isna().sum())
        if pd.api.types.is_numeric_dtype(data[col]):
            summary_data['Mean (Numeric Columns)'].append(round(data[col].mean(), 2))
        else:
            summary_data['Mean (Numeric Columns)'].append('N/A')
    summary_df = pd.DataFrame(summary_data)
    return summary_df

if __name__ == "__main__":
    # Load data
    uploaded_files_folder = "uploaded_files/uploaded_files"
    files = os.listdir(uploaded_files_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    
    if len(csv_files) == 0:
        st.warning("No CSV files found in the 'uploaded_files' folder.")
    else:
        # Automatically select the first CSV file found
        file_path = os.path.join(uploaded_files_folder, csv_files[0])
        data = pd.read_csv(file_path)

    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
        

    
    # Page title
    st.title('Data Visualization Before Cleaning')

    # Split page into two columns: Left for summary and selection, Right for visualizations
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.subheader('Data Summary')
        summary_df = display_data_summary(data)
        # Displaying DataFrame without scroll bars
        st.table(summary_df)
        
        # Dropdown for column selection, filtered for numeric columns for boxplot and all columns for histograms and pie charts
        selected_column = st.selectbox('Select a column for visualization:', data.columns)

    with right_column:
        if selected_column:
            # Histogram
            if pd.api.types.is_numeric_dtype(data[selected_column]) or data[selected_column].dtype == 'object':
                st.subheader('Histogram')
                hist_fig = plot_histograms(data, selected_column)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            # Boxplot for numeric data
            if pd.api.types.is_numeric_dtype(data[selected_column]):
                st.subheader('Boxplot')
                box_fig = plot_boxplot(data, selected_column)
                st.plotly_chart(box_fig, use_container_width=True)
            
            # Pie Chart for categorical data
            elif data[selected_column].dtype == 'object':
                st.subheader('Pie Chart')
                pie_fig = plot_pie_chart(data, selected_column)
                st.plotly_chart(pie_fig, use_container_width=True)
    
    if st.button("Clean Me"):
        switch_page("page3_CleanData")
