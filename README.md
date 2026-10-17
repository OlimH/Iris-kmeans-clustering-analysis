[README_Iris_KMeans.md](https://github.com/user-attachments/files/33587303/README_Iris_KMeans.md)
# Iris Flowering Plant — K-Means Clustering Analysis

An end-to-end exploratory **unsupervised machine learning** analysis of the Iris flower dataset using Python, scikit-learn, and interactive Plotly visualizations. The project examines whether four flower measurements reveal natural groupings, compares **K = 2** with **K = 3**, and evaluates the resulting clusters against known species labels.

## Project objective

The Iris dataset contains **150 observations** across three species (**50 each**): *Iris setosa*, *Iris versicolor*, and *Iris virginica*. Each flower has four numerical measurements in centimeters:

- Sepal length
- Sepal width
- Petal length
- Petal width

The goal is to discover clusters from these four measurements **without using species labels to fit K-Means**. The known labels are used afterward for external evaluation.

## Tools and libraries

- **Python** and **Jupyter Notebook** — analysis and documentation
- **pandas** and **NumPy** — data inspection and preparation
- **scikit-learn** — Iris dataset, StandardScaler, KMeans, PCA, train-test split, and evaluation metrics
- **Matplotlib** — Elbow Method and Silhouette Score plots
- **Plotly** — interactive boxplots and clustering scatterplots

## Workflow

### 1. Data loading and quality checks

Loaded the Iris dataset using `load_iris(as_frame=True)` and selected the four measurement columns for clustering. Inspected structure, missing values, duplicates, descriptive statistics, dimensions, data types, and nonpositive measurements.

**Findings:** 150 rows × 4 measurement columns; no missing values; one duplicate measurement row; no zero or negative measurements.

### 2. Outlier analysis

Used Plotly boxplots and the **1.5 × interquartile range (IQR)** rule to flag potential outliers. The notebook identifies potential outliers in **sepal width** and retains all observations because unusual measurements may reflect natural biological variation.

### 3. Feature scaling

Applied **Z-score standardization** using `StandardScaler` to the four numerical measurements. The standardized features have means approximately equal to **0** and population standard deviations of **1**, making their numerical scales more comparable for Euclidean-distance-based K-Means.

### 4. Selecting the number of clusters

Tested **K = 2 through K = 10** using:

- **Elbow Method:** within-cluster sum of squares (WCSS/inertia), looking for diminishing returns.
- **Silhouette Score:** how cohesive and separated the clusters are.

The elbow suggests **K = 3** as a reasonable candidate, while the highest Silhouette Score occurs at **K = 2**.

### 5. Compare K = 2 and K = 3

Fit K-Means on all **four standardized features**, then visualized the resulting assignments in two ways:

- **PCA scatterplots:** two principal components summarizing the four standardized measurements.
- **Petal length vs. petal width:** the original measurement axes, colored using cluster assignments from the four-feature models.

PCA was used **for visualization only**, not as the input to K-Means. K = 2 isolates Setosa-like flowers from the other two species; K = 3 reveals an additional group, with overlap between Versicolor and Virginica.

| Full-dataset metric | K = 2 | K = 3 |
|---|---:|---:|
| WCSS / inertia | 222.3617 | 139.8205 |
| Silhouette Score | **0.5818** | 0.4599 |
| Adjusted Rand Index (ARI) | 0.5681 | **0.6201** |
| Normalized Mutual Information (NMI) | **0.7337** | 0.6595 |

**Interpretation:** K = 2 offers clearer geometric separation and higher NMI. K = 3 achieves higher ARI against the three species labels, making it the preferred exploratory solution for investigating all three species. No single metric establishes that one solution is universally best.

### 6. External evaluation against actual species

Compared cluster assignments with the original target labels using cross-tabulation, **ARI**, and **NMI**. With K = 3, the notebook's full-dataset cross-tabulation shows:

| Actual species | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---:|---:|---:|
| Setosa | 0 | 50 | 0 |
| Versicolor | 39 | 0 | 11 |
| Virginica | 14 | 0 | 36 |

All 50 Setosa flowers form a distinct cluster. Under majority-species matching, **125 of 150 (83.3%)** observations agree with their species labels. This is an **in-sample descriptive agreement rate**, not supervised classification accuracy.

### 7. Train-test evaluation on unseen observations

Conducted a separate **stratified 80/20 split**: 120 training flowers and 30 test flowers, with four input features. To avoid leakage, the scaler was **fitted on training features only** and then applied to the test features. K-Means with **K = 3** was fitted on the training set; the model then assigned test flowers to the learned centroids.

| Held-out test metric | Result |
|---|---:|
| Adjusted Rand Index (ARI) | **0.5489** |
| Normalized Mutual Information (NMI) | **0.6293** |
| Silhouette Score | **0.3940** |

Test-set cross-tabulation:

| Actual species | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---:|---:|---:|
| Setosa | 0 | 10 | 0 |
| Versicolor | 3 | 0 | 7 |
| Virginica | 7 | 0 | 3 |

All 10 Setosa test flowers were isolated. Under majority-species matching, **24 of 30 (80%)** test observations agree with the species labels. This is a descriptive post-hoc agreement measure, **not a supervised classification accuracy claim**. The holdout scores are from one split and may vary with a different split.

## Key takeaways

- **Setosa is distinctly separable** from the other Iris species based on these measurements.
- **Versicolor and Virginica overlap**, making their separation more challenging.
- **K = 2** has the strongest Silhouette Score; **K = 3** has the stronger full-dataset ARI and supports analysis of all three species.
- Scaling matters for distance-based clustering, and **training-only scaling** is important in the held-out evaluation.
- The target labels were **not used to fit the K-Means models**; they were used only for evaluation and stratification of the train-test split.

## Repository contents

```text
iris-kmeans-clustering-analysis/
├── README.md
├── Iris Flowering Plant k-means Clustering Analysis.ipynb
├── iris.png                   # Add the image used by the notebook
└── requirements.txt           # Optional, for reproducibility
```

> The notebook references `iris.png` in its introductory Markdown. Add that image alongside the notebook so it renders on GitHub. The notebook loads Iris data directly from scikit-learn; no separate CSV is required.

## Run the notebook

1. Clone or download this repository.
2. Install the required Python packages:

   ```bash
   pip install numpy pandas scikit-learn matplotlib plotly jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open **`Iris Flowering Plant k-means Clustering Analysis.ipynb`** and run the cells in order.

**Note:** On some Windows/MKL installations, scikit-learn may display a K-Means threading warning. The warning does not by itself indicate that the clustering failed.

## Limitations

This is an **exploratory clustering project**, not a production species classifier. K-Means assumes distance-based, centroid-oriented groupings, and its results depend on feature scaling and the chosen K. ARI and NMI use known labels **after fitting** to evaluate alignment. PCA plots are two-dimensional projections and may hide structure visible in four dimensions. The held-out evaluation uses only one train-test split.

## Conclusion

The analysis shows that K-Means can discover meaningful structure in the Iris measurements. Two clusters provide stronger separation, while three clusters better represent the three species according to ARI, though Versicolor and Virginica overlap. The project demonstrates an end-to-end unsupervised learning workflow: data validation, outlier inspection, standardization, cluster selection, interactive visualization, external validation, and assignment of unseen observations.
