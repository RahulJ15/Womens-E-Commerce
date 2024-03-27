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
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        if num_clusters == 3:  # Display inferences only when k=3
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
        else:
            st.write(f"Graphs are displayed for {num_clusters} clusters.")
            # Scatter plot visualizations for specified pairs of features
            st.subheader("Scatter Plots by Cluster")
            
            # Scatter plot for Age vs. Rating
            fig_age_rating = px.scatter(data, x='Rating', y='Age', color='Cluster', 
                                        title="Rating vs. Age (Colored by Cluster)")
            st.plotly_chart(fig_age_rating)
            
            # Scatter plot for Age vs. Positive Feedback Count
            fig_age_positive_feedback = px.scatter(data, x='Age', y='Positive Feedback Count', color='Cluster', 
                                                   title="Age vs. Positive Feedback Count  (Colored by Cluster)")
            st.plotly_chart(fig_age_positive_feedback)
    
            # Scatter plot for Rating vs. Positive Feedback Count
            fig_rating_positive_feedback = px.scatter(data, x='Rating', y='Positive Feedback Count', color='Cluster', 
                                                      title="Rating vs. Positive Feedback Count (Colored by Cluster)")
            st.plotly_chart(fig_rating_positive_feedback)




elif model_type == "hierarchical clustering":
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

        #ok
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

        st.write("""
         - The dendrogram shows how individual or groups of customers are merged at different distances.
         - The height of the dendrogram branches indicates the Euclidean distances at which clusters were joined.
         - The color-coding of branches suggests that cluster 2 (blue branch) is quite distinct, while clusters 0 and 1 (red and green branches) are more similar to each other.
         """)

        # Count the number of data points in each cluster
        cluster_counts = pd.Series(data['Cluster']).value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']

        # Create Plotly Express bar plot
        fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Distribution of Clusters (Interactive)',
             labels={'Cluster': 'Cluster', 'Count': 'Count'}, color='Cluster')

       # Update layout
        fig.update_layout(
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Count'),
        hovermode='x',  # Display hover information along the x-axis
)

       # Show figure
        st.plotly_chart(fig)

        st.write("""
    - The bar chart provides a count of data points in each cluster.
    - Cluster 0 has the most data points, indicating that a large portion of the data falls into this cluster.
    - Cluster 1 has significantly fewer data points, and cluster 2 has the least, suggesting a potential outlier group or less common data profiles.
""")

# Scatter plot for two principal components (2D)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = data['Cluster']

        fig_scatter_2d = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='2D Scatter Plot of Clusters (PCA)',
                            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'Cluster': 'Cluster'})

# Add hover information
        fig_scatter_2d.update_traces(marker=dict(size=8),
                             selector=dict(mode='markers'))

# Show 2D scatter plot
        st.plotly_chart(fig_scatter_2d)

        st.write("""
    - Data is visualized in two dimensions using PCA, which is helpful for observing the spread and overlap of clusters.
    - The clusters are color-coded, showing a gradient from cluster 0 to cluster 2.
    - The separation between clusters is visible but there are regions where data points overlap, particularly between clusters 0 and 1.
""")

# Scatter plot for three principal components (3D)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features_scaled)
        pca_df_3d = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
        pca_df_3d['Cluster'] = data['Cluster']

        fig_scatter_3d = px.scatter_3d(pca_df_3d, x='PC1', y='PC2', z='PC3', color='Cluster', title='3D Scatter Plot of Clusters (PCA)',
                               labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3', 'Cluster': 'Cluster'})

# Add hover information
        fig_scatter_3d.update_traces(marker=dict(size=4),
                              selector=dict(mode='markers'))

# Show 3D scatter plot
        st.plotly_chart(fig_scatter_3d)

        st.write("""
    - The 3D visualization provides a more detailed view of the clusters' separations and densities.
    - There is a concentration of data points in cluster 0, suggesting tight grouping.
    - Clusters 1 and 2 are more spread out, with cluster 2 data points higher on the Principal Component 3 axis, indicating a different variance direction.
""")


    
    st.write("""
            -Marketing Stratergy Analysis-


        Customer Segmentation:
        - Customers are segmented into three distinct groups, potentially based on purchasing behavior, demographics, or product preferences.
        - Segment 0 represents the largest customer base, suggesting a general marketing strategy with broad appeal.
        - Segment 1 is smaller, which could indicate a niche market that may respond to more specialized marketing campaigns.
        - Segment 2, being the smallest, might represent a premium or atypical segment that requires unique marketing approaches, possibly higher-value customers with specific needs.

        Targeted Marketing:
        - The PCA plots suggest different variance in customer characteristics; marketing can tailor messages to highlight features or products that resonate with each segment's interests.
        - Overlaps in the 2D PCA plot imply some shared interests between segments 0 and 1, hinting at the potential for cross-selling strategies.
        - The distinct separation of segment 2 in the 3D plot suggests unique traits that could be leveraged for highly targeted marketing.

        Resource Allocation:
        - Given the size of segment 0, more resources could be allocated to target this group for general marketing campaigns aimed at volume sales.
        - Segments 1 and 2 may require more personalized engagement strategies, possibly requiring a higher investment per customer but potentially yielding higher margins.

        Product Development:
        - Insights from clustering can guide product development to better serve the identified segments, focusing on features and services valued by each group.
        - The differences in clusters may reflect different usage patterns or needs, guiding the development of customized products.

        Customer Retention:
        - By understanding the characteristics of each cluster, retention strategies can be tailored to address the specific desires or pain points of each segment.
        - Smaller clusters may indicate customers at risk of churn who could benefit from targeted retention programs.

        Market Positioning:
        - If clusters align with different product lines or services, this can inform how to position these offerings in the market effectively.
        - The unique characteristics of segment 2 could guide premium positioning or the introduction of loyalty programs.

                """)

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

        color_scale = px.colors.sequential.Blues

        # Create the 3D scatter plot with the custom color scale
        fig_3d_dbscan = px.scatter_3d(data, x='Rating', y='Age', z='Positive Feedback Count', color='Cluster',
                                    title="3D Scatter Plot (Rating, Age, Positive Feedback Count)",
                                    color_continuous_scale=color_scale)
        # Display the plot
        st.plotly_chart(fig_3d_dbscan)

        st.subheader("Actionable Marketing Insights from DBSCAN Clustering")
        st.write("""
            - Inference:
            Based on the provided outputs for different combinations of `eps` and `min_samples`:
            -The choice of `min_samples` significantly affects the number of clusters and noise points identified by the DBSCAN algorithm. As the `min_samples` parameter increases, the number of clusters tends to decrease while the number of noise points tends to increase.
            - The top 3 combinations based on the provided outputs are as follows:
            1. For eps=1 and min_samples=1:
            - This combination yields the highest number of clusters (226) without any noise points. It suggests a fine granularity in clustering, capturing a large number of distinct groups in the data.
                  Given the spread of points at each rating level, there doesn’t seem to be a strong, consistent trend where a specific age group favors a certain rating. All rating levels have a wide range of ages represented.
                 The presence of isolated points at higher ages for certain ratings might indicate outliers or unique cases within those rating groups.
                 
            2. For eps=1.5 and min_samples=1:
            - This combination results in a substantial number of clusters (48) with no noise points. It indicates a slightly lower granularity compared to the first combination but still captures a significant level of detail in the data.
                 Given the slight increase in `eps`, we might start to see some small clusters forming where data points are within 1.5 units of age and have the same rating.
                 The presence of any clusters would suggest that there are groups of individuals with similar ages and ratings or feedback counts that are within a close range of each other.

                 If the clusters are still highly individualized (each point is its own cluster), it would suggest that the dataset has a lot of variability, and the individuals' ages and ratings or feedback counts do not group into tight clusters within the 1.5 range.

                 If no clusters are observed, it could imply that even with a slightly larger `eps`, the data points do not naturally group together, which could indicate a high level of diversity in the dataset

            3. For eps=2 and min_samples=1:
            - This combination produces fewer clusters (15) compared to the previous two combinations but still retains a clean clustering structure without any noise points. It represents a more generalized clustering approach, capturing broader patterns in the data.
                A higher density of data points could potentially lead to some clustering at lower ages and feedback counts, as these points are closer together.
                we begin to see larger clusters forming at lower feedback counts and younger ages, it may indicate that younger individuals are more active in giving feedback, or that the majority of feedback comes from this demographic.  """)

if st.button("Click for Sentiment Analysis"):
    switch_page("page5_SentimentAnalysis")

