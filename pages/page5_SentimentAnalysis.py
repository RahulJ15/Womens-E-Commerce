import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import streamlit as st
import streamlit as st
from streamlit_custom_notification_box import custom_notification_box

# Set page config
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Page header
st.title("Sentiment Analysis on Customer Reviews")

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
    
    # Convert the Review Text column to string and calculate sentiment polarity
    data['Review Text'] = data['Review Text'].astype(str)
    data['polarity'] = data['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Drop rows with missing values and reset index
    data = data.dropna().reset_index(drop=True)

    # Arrange the Plotly visualizations in two columns
    col1, col2 = st.columns(2)

    with col1:
        # Visualization 1: Count of Ratings
        fig_ratings = px.histogram(data, x='Rating', title='Count of Ratings')
        fig_ratings.update_layout(width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_ratings)
        st.write("""
        - A high number of 5-star ratings, suggesting that customers tend to leave reviews when they are highly satisfied. 
        - This might also reflect a positive overall customer sentiment towards the products.
        - Ratings of 4 and 5 are significantly more common than 1, 2, or 3, which might imply good product quality or customer service overall.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Showcase top reviews in ads to highlight customer love & Run a 'Share Your Thoughts' campaign to boost high-rated categories.""")


        # Visualization 3: Age vs Positive Feedback
        fig_age_feedback = px.scatter(data, x='Age', y='Positive Feedback Count', 
                                      title='Age vs Positive Feedback Count', trendline='ols', 
                                      color='Age', size='Positive Feedback Count', hover_data=['Class Name'])
        fig_age_feedback.update_layout(width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_age_feedback)
        st.write("""
        - There's a wide distribution of ages providing positive feedback, but there appears to be a concentration of feedback from customers in the 30-50 age range. 
        - This could suggest that this demographic is more engaged in providing feedback or they may represent a larger segment of the customer base.
        - Very high positive feedback counts are more scarce, possibly indicating that most customers don’t leave a lot of feedback unless exceptionally motivated.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Craft campaigns that resonate with the 30-50 sweet spot & Pitch quality and style to our most engaged age bracket""")

        # Visualization 5: Boxplot of Polarity by Department Name
        fig_polarity_department = px.box(data, x='Department Name', y='polarity', 
                                         title='Polarity by Department Name', color='Department Name')
        fig_polarity_department.update_layout(width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_polarity_department)
        st.write("""
        - Sentiment polarity for different departments, with most departments having median polarities above zero, reinforcing the idea of generally positive reviews.
        - Some departments have a wider spread of sentiment, indicated by the larger interquartile ranges, suggesting a more varied customer experience.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Analyze sentiment by department to refine our offerings & Use varied feedback to steer product and service enhancements.""")

    with col2:
        # Visualization 2: Count of Reviews by Class Name
        fig_class_name = px.histogram(data, x='Class Name', title='Count of Reviews by Class Name', color='Class Name')
        fig_class_name.update_layout(width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_class_name)
        st.write("""
        - Certain classes of items, like Dresses and Knits, receive more reviews. 
        - This could indicate either a higher sales volume for these items or that they elicit more feedback from customers.
        - Less reviewed classes might either be less sold or less likely to inspire reviews, which could be due to either satisfaction or indifference.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Ramp up marketing for Dresses & Knits – our crowd pleasers! Delve into low-review categories for a revamp.""")

        # Visualization 4: Distribution of Polarity
        fig_polarity_dist = px.histogram(data, x='polarity', title='Distribution of Polarity', nbins=40)
        fig_polarity_dist.update_layout(width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_polarity_dist)
        st.write("""
        - The distribution is centered around a positive polarity, indicating that reviews are generally positive. 
        - There are very few reviews with strongly negative sentiments.
        - The skew towards positive polarity suggests that customers who leave reviews have a favorable view of the items they've purchased.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Launch a 'Review & Reward' program to inspire detailed feedback & Incentivize comprehensive reviews for rare insights.""")

        # Visualization 6: Pie Chart for Polarity Distribution
        polarity_counts = data['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative')).value_counts()
        fig_polarity_pie = go.Figure(data=[go.Pie(labels=polarity_counts.index, values=polarity_counts, hole=.3)])
        fig_polarity_pie.update_layout(title_text='Polarity Distribution', width=400, height=350)  # Adjust size here
        st.plotly_chart(fig_polarity_pie)
        st.write("""
        - Overwhelming majority of positive sentiment, confirming the conclusions drawn from the polarity distribution. 
        - This demonstrates a well-received product line or customer satisfaction with their purchases.
        - Negative sentiments represent a small fraction, but they cannot be overlooked as they provide opportunities for improvement in specific areas.
            """)
        st.write("""WHAT ACTION CAN BE TAKEN?""")
        st.write("""Capitalize on positivity in our promotions to lure in new customers & Transform negative feedback into improvement blueprints.""")
