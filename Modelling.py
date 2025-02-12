import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
import os
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Load the dataset
def load_data_from_uploaded_files_folder():
    uploaded_files_folder = "uploaded_files/uploaded_files"
    files = os.listdir(uploaded_files_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    
    if len(csv_files) == 0:
        st.warning("No CSV files found in the 'uploaded_files' folder.")
        return None
    else:
        # Automatically select the first CSV file found
        file_path = os.path.join(uploaded_files_folder, csv_files[0])
        return pd.read_csv(file_path)

# Load the dataset
data = load_data_from_uploaded_files_folder()

# Title for model selection
st.title("Model Selection")

# Dropdown menu for model selection
model_type = st.selectbox(
    "Choose a clustering model:",
    ("k-means", "hierarchical clustering", "DBSCAN")
)

if model_type == "k-means":
    # K-Means Clustering Visualization and Parameter Selection
    st.header("K-Means Clustering")
    
    # Elbow Method for Optimal k
    st.subheader("Elbow Method for Optimal k")
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data.select_dtypes(include=[np.number]))
        sse.append(kmeans.inertia_)
    
    # Plotting the Elbow Method graph
    fig_elbow = px.line(x=range(1, 11), y=sse, markers=True, title="Elbow Method Graph")
    fig_elbow.update_layout(xaxis_title="Number of Clusters", yaxis_title="Sum of Squared Distances", xaxis_dtick=1)
    st.plotly_chart(fig_elbow)
    
    # Description below elbow graph
    st.write("Optimal clusters for the dataset are identified as 3, where further increase in k yields minimal improvement in the sum of squared distances.")
    
    # Allow the user to select the number of clusters after viewing the elbow plot
    num_clusters = st.slider("Select the number of clusters (k):", min_value=2, max_value=10, value=3, step=1)
    
    # Perform K-Means Clustering and display results
    if st.button("Perform Clustering"):
        # Standardizing the features
        features = data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply KMeans and predict clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        data['Cluster'] = kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, data['Cluster'])
        st.write(f"Silhouette Score for {num_clusters} clusters:", silhouette_avg)
        st.write("Silhouette Score is moderate (~0.23), suggesting that while there is some cluster cohesion.")
        
        # Select only numeric columns for aggregation
        numeric_columns = data.select_dtypes(include=[np.number])

        # Perform aggregation on numeric columns
        cluster_stats = numeric_columns.groupby('Cluster').agg(['mean', 'std']).reset_index()

        
        # Renaming columns for a cleaner look
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        
        # Clean up the header by removing the unnecessary '_mean' and '_std' from the index column
        cluster_stats.rename(columns=lambda x: x.replace('_mean', '').replace('_std', '') if 'Cluster_' in x else x, inplace=True)
        
        # Split the statistics into mean and standard deviation DataFrames for better visual display
        cluster_mean_stats = cluster_stats[[col for col in cluster_stats.columns if '_mean' in col or 'Cluster' in col]]
        cluster_std_stats = cluster_stats[[col for col in cluster_stats.columns if '_std' in col or 'Cluster' in col]]
        
        # Display the mean statistics
        st.subheader("Mean Statistics by Cluster")
        st.dataframe(cluster_mean_stats)
        
        # Display the standard deviation statistics
        st.subheader("Standard Deviation Statistics by Cluster")
        st.dataframe(cluster_std_stats)
        
        # Description after cluster statistics tables
        st.write("""
        - **Cluster 0:** Low ratings, not recommended often, diverse feedback.
        - **Cluster 1:** High ratings, highly recommended, consistent feedback.
        - **Cluster 2:** High ratings like Cluster 1, slightly more varied feedback.
        - **Age:** Not a key differentiator across clusters.
        """)
        
        # Scatter plot visualizations for specified pairs of features
        st.subheader("Scatter Plots by Cluster")
        
        # Scatter plot for Age vs. Rating
        fig_age_rating = px.scatter(data, x='Rating', y='Age', color='Cluster', 
                                    title="Rating vs. Age (Colored by Cluster)")
        st.plotly_chart(fig_age_rating)
        
        st.write("""
                - Ratings are consistently high across all age groups for Clusters 1 and 2.
                - Cluster 0 shows a wider spread of ages at lower ratings
        """)
        
        # Scatter plot for Age vs. Positive Feedback Count
        fig_age_positive_feedback = px.scatter(data, x='Age', y='Positive Feedback Count', color='Cluster', 
                                               title="Age vs. Positive Feedback Count  (Colored by Cluster)")
        st.plotly_chart(fig_age_positive_feedback)

        st.write("""
                - Most of the positive feedback is given by a younger demographic across all clusters.
                - Older age groups provide relatively less feedback.
        """)

        # Scatter plot for Rating vs. Positive Feedback Count
        fig_rating_positive_feedback = px.scatter(data, x='Rating', y='Positive Feedback Count', color='Cluster', 
                                                  title="Rating vs. Positive Feedback Count (Colored by Cluster)")
        st.plotly_chart(fig_rating_positive_feedback)

        st.write("""
                - Higher ratings do not necessarily correspond to a higher positive feedback count.
                - Lower ratings (primarily in Cluster 0) have a wider range of feedback counts, suggesting variability in customer engagement.
        """)

        # Concluding with marketing insights derived from clustering
        st.subheader("Actionable Marketing Insights from K-Means Clustering")
        st.write("""
        - **Turnaround Plan for Cluster 0:** Dive deep into feedback, unveil the 'whys' of dissatisfaction, and launch targeted campaigns that say 'We've listened!'
        - **Amplify Voices from Clusters 1 & 2:** Celebrate the high-spirited reviews and recommendations with compelling stories for powerful word-of-mouth buzz.
        - **Energize the Youth Quotient:** Craft exclusive, youthful engagement programs that turn the feedback-rich younger demographic into trendsetting brand ambassadors.
        """)
        

elif model_type == "hierarchical clustering":
    import plotly.figure_factory as ff
    import plotly.express as px
    import matplotlib.pyplot as plt
    
    st.write("Hierarchical clustering model selected.")
    
    # Perform Hierarchical Clustering and display results
    if st.button("Perform Clustering"):
        # Standardizing the features
        features = data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
        data['Cluster'] = hc.fit_predict(features_scaled)
        
        # Calculate linkage matrix
        linkage_matrix = sch.linkage(features_scaled.T, method='ward')  # Transpose the DataFrame for correct orientation

        # Create dendrogram figure
        fig = ff.create_dendrogram(linkage_matrix, orientation='bottom')

        # Update layout
        fig.update_layout(
            title='Hierarchical Clustering Dendrogram',
            xaxis=dict(title='Customers'),
            yaxis=dict(title='Euclidean Distances'),
            hovermode='x',  # Display hover information along the x-axis
            hoverdistance=5,  # Distance threshold for hover labels
            showlegend=False,  # Hide legend
        )

        # Add custom hover labels for dendrogram branches
        fig.update_traces(hovertext=["Cluster " + str(i+1) for i in range(len(fig.data[0]['x']))])

        # Add color to dendrogram branches
        for i in range(len(fig.data)):
            fig.data[i].marker.color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]

        # Show figure
        st.plotly_chart(fig)


elif model_type == "DBSCAN":
    # DBSCAN Clustering Visualization and Parameter Selection
    st.header("DBSCAN Clustering")

    # Allow the user to specify parameters for DBSCAN
    eps = st.slider("Select the maximum distance between two samples for them to be considered as in the same neighborhood (eps):", min_value=0.5, max_value=2.0, value=0.5, step=0.5)
    min_samples = st.slider("Select the number of samples in a neighborhood for a point to be considered as a core point (min_samples):", min_value=1, max_value=20, value=5, step=10)

    # Perform DBSCAN Clustering and display results
    if st.button("Perform Clustering"):
        # Standardizing the features
        features = data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply DBSCAN and predict clusters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        data['Cluster'] = dbscan.fit_predict(features_scaled)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(data['Cluster'])) - (1 if -1 in data['Cluster'] else 0)
        n_noise_ = list(data['Cluster']).count(-1)

        # Display cluster information
        st.write(f"Estimated number of clusters: {n_clusters_}")
        st.write(f"Estimated number of noise points: {n_noise_}")

        # Scatter plot visualizations for specified pairs of features
        st.subheader("Scatter Plots by Cluster")

        # Scatter plot for Age vs. Rating (2D)
        fig_age_rating_dbscan_2d = px.scatter(data, x='Rating', y='Age', color='Cluster',
                                              title="Rating vs. Age (Colored by Cluster)")
        st.plotly_chart(fig_age_rating_dbscan_2d)

        # Scatter plot for Age vs. Positive Feedback Count (2D)
        fig_age_positive_feedback_dbscan_2d = px.scatter(data, x='Age', y='Positive Feedback Count', color='Cluster',
                                                          title="Age vs. Positive Feedback Count (Colored by Cluster)")
        st.plotly_chart(fig_age_positive_feedback_dbscan_2d)

        # Scatter plot for Rating vs. Positive Feedback Count (2D)
        fig_rating_positive_feedback_dbscan_2d = px.scatter(data, x='Rating', y='Positive Feedback Count',
                                                             color='Cluster',
                                                             title="Rating vs. Positive Feedback Count (Colored by Cluster)")
        st.plotly_chart(fig_rating_positive_feedback_dbscan_2d)

        # 3D Scatter plot visualizations
        st.subheader("3D Scatter Plots by Cluster")

        # Scatter plot for Rating, Age, and Positive Feedback Count (3D)
        fig_3d_dbscan = px.scatter_3d(data, x='Rating', y='Age', z='Positive Feedback Count', color='Cluster',
                                      title="3D Scatter Plot (Rating, Age, Positive Feedback Count)")
        st.plotly_chart(fig_3d_dbscan)

        
        st.subheader("Actionable Marketing Insights from DBSCAN Clustering")
        st.write("""
            - Inference:
            Based on the provided outputs for different combinations of `eps` and `min_samples`:
            -The choice of `min_samples` significantly affects the number of clusters and noise points identified by the DBSCAN algorithm. As the `min_samples` parameter increases, the number of clusters tends to decrease while the number of noise points tends to increase.
            - The top 3 combinations based on the provided outputs are as follows:
            1. For eps=1 and min_samples=1:
            - This combination yields the highest number of clusters (226) without any noise points. It suggests a fine granularity in clustering, capturing a large number of distinct groups in the data.
            2. For eps=1.5 and min_samples=1:
            - This combination results in a substantial number of clusters (48) with no noise points. It indicates a slightly lower granularity compared to the first combination but still captures a significant level of detail in the data.
            3. For eps=2 and min_samples=1:
            - This combination produces fewer clusters (15) compared to the previous two combinations but still retains a clean clustering structure without any noise points. It represents a more generalized clustering approach, capturing broader patterns in the data.""")

if st.button("Click for Sentiment Analysis"):
    switch_page("page5_SentimentAnalysis")
