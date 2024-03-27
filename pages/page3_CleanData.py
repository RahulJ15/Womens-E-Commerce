import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
import os
# Function to plot histograms
def plot_histograms(data, column):
    fig = px.histogram(data, x=column, title=f"Histogram of {column}", 
                       color_discrete_sequence=['mediumslateblue'], 
                       barmode='overlay',  # Change barmode to overlay
                       barnorm='percent',  # Normalize the bars to show percentage
                       opacity=1,        # Set opacity for better visibility
                       nbins=20)           # Adjust the number of bins as needed
    fig.update_layout(bargap=0.1)  # Set the gap between bars
    return fig

# Function to plot pie chart
def plot_pie_chart(data, column):
    # Get top 5 categories
    top_5_categories = data[column].value_counts().nlargest(5).index.tolist()
    # Filter data for top 5 categories
    filtered_data = data[data[column].isin(top_5_categories)]
    # Plot pie chart
    fig = px.pie(filtered_data, names=column, title=f"Pie Chart of {column} (Top 5)",
                 hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# Function to display data summary
# Function to display data summary
def display_data_summary(data):
    summary_data = {
        'Column': [],
        'Data Type': [],
        'Unique Values': [],
        'Mean (Int Columns)': []
    }
    for col in data.columns:
        summary_data['Column'].append(col)
        summary_data['Data Type'].append(data[col].dtype)  # Get data type of column
        summary_data['Unique Values'].append(data[col].nunique())  # Get number of unique values
        if data[col].dtype == 'int64':  # Check if the column has integer data type
            summary_data['Mean (Int Columns)'].append(data[col].mean())  # Compute mean for integer columns
        else:
            summary_data['Mean (Int Columns)'].append(None)  # For non-integer columns, store None
    summary_df = pd.DataFrame(summary_data)
    
    # Header for count after dropping NA values
    st.write("### Count after dropping NA: 19,662")
    
    # Display data summary
    st.write(summary_df)



# Dictionary of inferences for each column
# Dictionary of inferences for each column
column_inferences = {
    'Class Name': "**The most sold item is Dresses.**",
    'Age': "**Age of customer mostly distributed in the range of 34 to 39.**",
    'Rating':'**Here, we can notice that 4 and 5 rates account about 77% of rating.**',
    'Recommended IND': '**About 82.2% products are recommended in the dataset.**',
    'Department Name':'**Younger people prefer to buy bottoms and dresses, while older people prefer to buy tops and jackets.**',
    # Add more inferences for other columns as needed
}



if __name__ == "__main__":
    # Load data from the uploaded_files folder
    uploaded_files_folder = "uploaded_files/uploaded_files"
    files = os.listdir(uploaded_files_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    
    if len(csv_files) == 0:
        st.warning("No CSV files found in the 'uploaded_files' folder.")
    else:
        # Automatically select the first CSV file found
        file_path = os.path.join(uploaded_files_folder, csv_files[0])
        data = pd.read_csv(file_path)
        
        # Drop 'Unnamed: 0' column
        if 'Unnamed: 0' in data.columns:
            data.drop(columns=['Unnamed: 0'], inplace=True)
        
    data = data.dropna().reset_index(drop=True)
    
    # Exclude specific columns
    excluded_columns = ['Unnamed: 0', 'Title', 'Review Text','Clothing ID','Positive Feedback Count']
    columns_to_display = [col for col in data.columns if col not in excluded_columns]
    
    # Title for the page
    st.title('Data Visualization After EDA')

    # Arrange layout using columns
    left_column, right_column = st.columns(2)

    # Left column: Display data summary and select column dropdown
    with left_column:
        st.subheader('Data Summary')
        display_data_summary(data)
        selected_column = st.selectbox('Select a column:', columns_to_display)
        st.subheader('Inference')
        if selected_column in column_inferences:
            st.markdown(f"**{column_inferences[selected_column]}**")

    # Right column: Display histogram, pie chart, and inference
    with right_column:
        st.subheader('Histogram')
        hist_fig = plot_histograms(data, selected_column)
        st.plotly_chart(hist_fig, use_container_width=True)
        
        st.subheader('Pie Chart')
        pie_fig = plot_pie_chart(data, selected_column)
        st.plotly_chart(pie_fig, use_container_width=True)

    if st.button("Go to Modelling"):
        switch_page("page4_Modelling")

