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

        # Convert 'Cluster' column to integer type
        data['Cluster'] = data['Cluster'].astype(int)
        
        # Calculate linkage matrix
        linkage_matrix = sch.linkage(features_scaled, method='ward')

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
        cluster_counts = data['Cluster'].value_counts().reset_index()
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
        fig_scatter_2d.update_traces(marker=dict(size=8), selector=dict(mode='markers'))

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
        fig_scatter_3d.update_traces(marker=dict(size=4), selector=dict(mode='markers'))

        # Show 3D scatter plot
        st.plotly_chart(fig_scatter_3d)

        st.write("""
        - The 3D visualization provides a more detailed view of the clusters' separations and densities.
        - There is a concentration of data points in cluster 0, suggesting tight grouping.
        - Clusters 1 and 2 are more spread out, with cluster 2 data points higher on the Principal Component 3 axis, indicating a different variance direction.
        """)

        st.subheader("Actionable Marketing Insights from Hierarchical Clustering")
        st.write("""
        - Customer Segmentation:
          - Customers are segmented into three distinct groups, potentially based on purchasing behavior, demographics, or product preferences.
          - Segment 0 represents the largest customer base, suggesting a general marketing strategy with broad appeal.
          - Segment 1 is smaller, which could indicate a niche market that may respond to more specialized marketing campaigns.
          - Segment 2, being the smallest, might represent a premium or atypical segment that requires unique marketing approaches, possibly higher-value customers with specific needs.
        
        - Targeted Marketing:
          - The PCA plots suggest different variance in customer characteristics; marketing can tailor messages to highlight features or products that resonate with each segment's interests.
          - Overlaps in the 2D PCA plot imply some shared interests between segments 0 and 1, hinting at the potential for cross-selling strategies.
          - The distinct separation of segment 2 in the 3D plot suggests unique traits that could be leveraged for highly targeted marketing.
        
        - Resource Allocation:
          - Given the size of segment 0, more resources could be allocated to target this group for general marketing campaigns aimed at volume sales.
          - Segments 1 and 2 may require more personalized engagement strategies, possibly requiring a higher investment per customer but potentially yielding higher margins.
        
        - Product Development:
          - Insights from clustering can guide product development to better serve the identified segments, focusing on features and services valued by each group.
          - The differences in clusters may reflect different usage patterns or needs, guiding the development of customized products.
        
        - Customer Retention:
          - By understanding the characteristics of each cluster, retention strategies can be tailored to address the specific desires or pain points of each segment.
          - Smaller clusters may indicate customers at risk of churn who could benefit from targeted retention programs.
        
        - Market Positioning:
          - If clusters align with different product lines or services, this can inform how to position these offerings in the market effectively.
          - The unique characteristics of segment 2 could guide premium positioning or the introduction of loyalty programs.
        """)