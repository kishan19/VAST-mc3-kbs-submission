# Topic Clustering with OpenAI Embeddings
# Convert text content to embeddings and perform topic clustering to identify groups of entities

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from secret/.env
load_dotenv(dotenv_path="secret/.env")


api_key = os.getenv("OPENAI_API_KEY")
# Set your OpenAI API key (ensure you have it in your environment or replace with your key)
client = OpenAI(api_key=api_key)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def create_embeddings_and_cluster(df, num_topics=10):
    """
    Convert text content to embeddings and perform topic clustering
    """
    print("Creating embeddings for text content...")
    
    # Get all content from the dataframe
    content_list = df['Processed_Text'].tolist()
    
    # Create embeddings for all content
    embeddings_list = embeddings.embed_documents(content_list)
    embeddings_array = np.array(embeddings_list)
    
    print(f"Created embeddings for {len(content_list)} messages")
    print(f"Embedding dimensions: {embeddings_array.shape}")
    
    # Perform K-means clustering
    print(f"Performing K-means clustering with {num_topics} topics...")
    kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Find representative messages for each cluster (closest to center)
    representative_messages = []
    for i in range(num_topics):
        cluster_mask = cluster_labels == i
        cluster_embeddings = embeddings_array[cluster_mask]
        cluster_center = cluster_centers[i]
        
        # Calculate distances to center
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get the actual index in the original dataframe
        cluster_indices = np.where(cluster_mask)[0]
        representative_idx = cluster_indices[closest_idx]
        representative_messages.append({
            'cluster': i,
            'message': content_list[representative_idx],
            'source_entity': df.iloc[representative_idx]['source_entity'],
            'target_entity': df.iloc[representative_idx]['target_entity']
        })
    
    return df_with_clusters, cluster_labels, cluster_centers, representative_messages

def analyze_entity_topics(df_with_clusters):
    """
    Analyze which entities are most active in each topic cluster
    """
    print("Analyzing entity participation in topics...")
    
    # Group by cluster and analyze entities
    cluster_entity_analysis = {}
    
    for cluster_id in range(10):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Count source entities
        source_entities = cluster_data['source_entity'].value_counts()
        target_entities = cluster_data['target_entity'].value_counts()
        
        # Combine all entities
        all_entities = pd.concat([source_entities, target_entities]).groupby(level=0).sum()
        all_entities = all_entities.sort_values(ascending=False)
        
        cluster_entity_analysis[cluster_id] = {
            'total_messages': len(cluster_data),
            'top_entities': all_entities.head(10).to_dict(),
            'source_entities': source_entities.head(5).to_dict(),
            'target_entities': target_entities.head(5).to_dict()
        }
    
    return cluster_entity_analysis

def generate_topic_summaries(representative_messages):
    """
    Generate topic summaries using GPT-4
    """
    print("Generating topic summaries...")
    
    topic_summaries = []
    
    for rep_msg in representative_messages:
        prompt = f"""
        Analyze the following message and provide a concise topic summary (2-3 words) that captures the main theme:
        
        Message: {rep_msg['message']}
        Source: {rep_msg['source_entity']}
        Target: {rep_msg['target_entity']}
        
        Provide only the topic summary, nothing else.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            topic_summary = response.choices[0].message.content.strip()
        except Exception as e:
            topic_summary = f"Topic {rep_msg['cluster']}"
        
        topic_summaries.append({
            'cluster': rep_msg['cluster'],
            'summary': topic_summary,
            'representative_message': rep_msg['message']
        })
    
    return topic_summaries

def visualize_clusters(embeddings_array, cluster_labels, topic_summaries):
    """
    Visualize the clusters using PCA
    """
    print("Creating cluster visualization...")
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    for i in range(10):
        cluster_mask = cluster_labels == i
        plt.scatter(
            embeddings_2d[cluster_mask, 0], 
            embeddings_2d[cluster_mask, 1], 
            c=colors[i], 
            label=f"Topic {i+1}: {topic_summaries[i]['summary']}",
            alpha=0.6
        )
    
    plt.title('Topic Clusters of Communication Messages')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('topic_clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_data_from_neo4j():
    """
    Load data from Neo4j database
    """

    # df = graph.run(query).to_data_frame()
    #df=pd.read_csv('data/MC3_comms_data_final.csv')
    df=pd.read_csv('data/MC3_data_no_pseudonyms.csv')
    df['source_entity']=df['source'].str.lower()
    df['target_entity']=df['target'].str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def main():
    """
    Main function to execute the topic clustering analysis
    """
    print("Starting topic clustering analysis...")
    
    # Load data
    df = load_data_from_neo4j()
    if df is None:
        print("Failed to load data from Neo4j. Please ensure the database is running.")
        return
    
    print(f"Loaded {len(df)} messages from database")
    
    # Execute the analysis
    df_with_clusters, cluster_labels, cluster_centers, representative_messages = create_embeddings_and_cluster(df, num_topics=10)

    # Analyze entity participation
    cluster_entity_analysis = analyze_entity_topics(df_with_clusters)

    # Generate topic summaries
    topic_summaries = generate_topic_summaries(representative_messages)

    # Create embeddings array for visualization
    content_list = df['content'].tolist()
    embeddings_list = embeddings.embed_documents(content_list)
    embeddings_array = np.array(embeddings_list)

    # Visualize clusters
    visualize_clusters(embeddings_array, cluster_labels, topic_summaries)

    # Display results
    print("\n" + "="*80)
    print("TOPIC CLUSTERING RESULTS")
    print("="*80)

    for i, summary in enumerate(topic_summaries):
        print(f"\nTopic {i+1}: {summary['summary']}")
        print(f"Representative Message: {summary['representative_message'][:100]}...")
        print(f"Total Messages: {cluster_entity_analysis[i]['total_messages']}")
        print("Top Entities:")
        for entity, count in list(cluster_entity_analysis[i]['top_entities'].items())[:5]:
            print(f"  - {entity}: {count} messages")
        print("-" * 60)

    # Save results to file
    results = {
        'topic_summaries': topic_summaries,
        'cluster_entity_analysis': cluster_entity_analysis,
        'representative_messages': representative_messages,
        'cluster_distribution': np.bincount(cluster_labels).tolist()
    }

    with open('data/topic_clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to 'topic_clustering_results.json'")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    # Save clustered dataframe
    df_with_clusters.to_csv('data/clustered_messages.csv', index=False)
    print("Clustered messages saved to 'clustered_messages.csv'")

if __name__ == "__main__":
    main() 