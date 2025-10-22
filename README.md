# Wine Quality Clustering

## Application link: https://huggingface.co/spaces/Nishal07/Wine_Dataset-Clustering

## Overview

This project performs unsupervised machine learning clustering analysis on wine quality data to identify distinct groups of wines based on their physicochemical properties. The goal is to discover natural groupings in the data that can provide insights into wine characteristics and quality.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Clustering Results](#clustering-results)
- [Algorithm Selection](#algorithm-selection)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
git clone https://github.com/NISHAL2007/Wine_Quality_Clustering.git
cd Wine_Quality_Clustering
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage

### Basic Usage

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('wine_quality.csv')

# Preprocess and scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('quality', axis=1))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

### Running the Notebook

```bash
jupyter notebook Wine_Quality_Clustering.ipynb
```

## Dataset

The dataset contains physicochemical properties of red and white wines:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Clustering Results

### Silhouette Scores

The silhouette score measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to 1, where higher values indicate better-defined clusters.

| Number of Clusters | Silhouette Score | Davies-Bouldin Index |
|-------------------|------------------|---------------------|
| 2                 | 0.342            | 1.245               |
| 3                 | 0.389            | 1.102               |
| 4                 | 0.356            | 1.187               |
| 5                 | 0.328            | 1.298               |
| 6                 | 0.315            | 1.354               |

**Optimal Number of Clusters: 3** (highest silhouette score and lowest Davies-Bouldin index)

Elbow Method and Silhouette Analysis
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/ab358b37-8260-439e-8c07-4fb6dde54304" />


### Cluster Characteristics

- **Cluster 0**: High alcohol content, lower acidity
- **Cluster 1**: Moderate alcohol, balanced acidity
- **Cluster 2**: Lower alcohol, higher acidity


## Algorithm Selection

### Why KMeans?

We selected **KMeans** as the primary clustering algorithm for the following reasons:

1. **Scalability**: KMeans efficiently handles the wine dataset size with O(n*k*i) complexity, where n is samples, k is clusters, and i is iterations.

2. **Interpretability**: The algorithm produces clear, spherical clusters with well-defined centroids that represent "typical" wine profiles for each group.

3. **Numerical Data Compatibility**: Wine quality features are continuous numerical variables, which align perfectly with KMeans' distance-based approach.

4. **Performance**: KMeans converges quickly and provides consistent results with the random_state parameter, making it reproducible.

5. **Validation Metrics**: Strong silhouette scores (0.389 for k=3) indicate well-separated, cohesive clusters.

6. **Feature Space**: After standardization, the wine features exhibit relatively uniform variance, making Euclidean distance an appropriate similarity metric.

### Comparison with Alternatives

- **DBSCAN**: Rejected due to difficulty in parameter tuning and lack of clear density-based clusters in the feature space.
- **Hierarchical Clustering**: Computationally expensive for this dataset size and doesn't provide significant advantages.
- **Gaussian Mixture Models**: More complex than needed; KMeans' hard clustering is sufficient for wine categorization.

## Visualizations

The project includes several visualizations:

- Elbow method plot for optimal k selection
- Silhouette analysis plots
- 2D PCA projections of clusters
- Feature distribution across clusters
- Correlation heatmaps

PCA Visualization:
<img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/1e64208e-dd9e-41b6-91c6-b64606006053" />


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

Nishal A T

## Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Inspired by wine quality research and clustering techniques
