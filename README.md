# Iris Flower Clustering Analysis Using K-Means

An end-to-end **unsupervised machine learning** project exploring natural groupings in the Iris dataset using Python, K-Means clustering, PCA, and interactive Plotly visualizations.

## Project Overview

The Iris dataset contains 150 flower samples from three species: **Setosa, Versicolor, and Virginica** (50 samples each). Each sample has four numerical measurements in centimeters: sepal length, sepal width, petal length, and petal width.

**Objective:** Discover clusters based on the four flower measurements without using species labels during model fitting, then compare the resulting groups with the known species as an external evaluation.

## Tools and Technologies

- Python, Jupyter Notebook
- pandas, NumPy
- scikit-learn (StandardScaler, KMeans, PCA, evaluation metrics)
- Plotly and Matplotlib

## Workflow

1. **Data inspection:** Review dimensions, types, missing values, duplicates, summary statistics, and plausible measurement ranges.
2. **Outlier exploration:** Inspect boxplots and IQR-based outlier flags; retain plausible observations.
3. **Feature scaling:** Standardize the four measurements using Z-score scaling.
4. **Cluster selection:** Compare candidate K values using the Elbow Method (WCSS/inertia) and Silhouette Score.
5. **Clustering:** Fit and compare K-Means models with **K = 2** and **K = 3**.
6. **Visualization:** Display clusters using PCA (two components) and original petal length versus petal width. Both visualizations show cluster labels learned from all four standardized measurements.
7. **External evaluation:** Compare cluster assignments against the known species using cross-tabulations, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).
8. **Holdout evaluation:** Perform a stratified 80/20 train-test split; fit the scaler and K-Means on training data only, and assign unseen test observations to the learned centroids.

## Key Findings

### Full-dataset comparison

| Metric | K = 2 | K = 3 |
|---|---:|---:|
| Silhouette Score (approx.) | 0.58 | 0.46 |
| Adjusted Rand Index | 0.5681 | 0.6201 |
| Normalized Mutual Information | 0.7337 | 0.6595 |

- **K = 2** separated Setosa from the other species more clearly and achieved the higher Silhouette Score.
- **K = 3** distinguished three groups, but Versicolor and Virginica overlapped. It achieved the higher ARI and was selected for the species-comparison exercise.
- With K = 3, the majority-species mapping agreed with **125 of 150** observations (83.3%). This is descriptive cluster-to-label agreement, **not** supervised predictive accuracy.

### Holdout evaluation (K = 3)

The dataset was split into **120 training** and **30 testing** observations. Only the original **four measurements** were used as model features.

| Test metric | Result |
|---|---:|
| Adjusted Rand Index | 0.5489 |
| Normalized Mutual Information | 0.6293 |
| Silhouette Score | 0.3940 |

On the test set, Setosa formed a distinct group. Versicolor and Virginica remained partly overlapping. After matching clusters to their majority species, **24 of 30** test observations (80%) agreed with their species labels. This is descriptive agreement, not supervised classification accuracy. Results correspond to the notebook run with `random_state=42` and may vary with code or environment changes.

## Visualizations

### Elbow Method and Silhouette Score

![Elbow Method and Silhouette Score](images/elbow_silhouette.png)

### Cluster Comparison (K = 2 vs K = 3)

![K-Means clustering visualizations](images/clustering_comparison.png)

> Add the actual exported plots to the `images/` folder with these filenames before publishing, or remove image links until the files are available.

## How to Run

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `Iris_KMeans_Clustering.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells in order from top to bottom.

The dataset is loaded from `sklearn.datasets.load_iris`, so no separate dataset download is required.

## Conclusion

K-Means identified meaningful structure in the Iris measurements, particularly the distinct Setosa group. Although K = 2 produced better-separated clusters, K = 3 aligned more closely with the three known species according to ARI. Overlap between Versicolor and Virginica illustrates the limitations of distance-based clustering. The holdout analysis demonstrates assigning unseen observations to learned cluster centroids without using species labels for training.

## Limitations

- Cluster IDs are arbitrary and do not directly represent species names.
- PCA plots compress four features into two dimensions and can hide some separation.
- The Elbow Method does not always identify a unique best K.
- A single small holdout split is illustrative, not a robust estimate of future performance.

## Dataset Credit

The classic Iris dataset, originally associated with Ronald A. Fisher's 1936 work, is accessed through scikit-learn's built-in dataset loader.
