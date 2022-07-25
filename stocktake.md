## what does this do?
algorithms
- clustering
    - kmeans
    - dbscan
    - EM with GMM
    - HOG extraction(?)
- parameter estimation
- benchmarking?
    - [here](/../Jupyter-Notebooks/Toolbox/clustering_tools.py)

- other
    - PCA data visualization (this is possibly not the actual algorithm name)

## sensible organisation structures
### by algorithm 
- <algo name>
    - quick example
    - parameter estimation
    - benchmarking (whatever the fuck that is. is that a thing?)
#### e.g.
- PCA
    - what it's useful for?
    - is there parameter estimation?
- DBSCAN
    - ...
- KMeans
    - ...
- EM with GMM
    - ...


data
 - [why is this important?]

## Folder Structure
- Notes

    ---


- Publishing

- Jupyter Notebooks
    - ---
    - Iris Clustering-Working.ipynb
        ```
        cluster iris data
            - with
                - kmeans
                - dbscan
                - EM with GMM
            - plot results
            
            - parameter estimation
                - kmeans
                - dbscan

            - visualize using PCA

            - using:
                - plotting tools, and
                - clustering tools
        ```
    - debgug_kmeans_plotting.py
    - debug_DBSCAN_plotting.py
    - Wine Clustering-Working.ipynb
    - Toolbox/
        - ---
        - clustering_tools.py
        - plotting_tools.py
        - ---
    - ---

  
- Scripts/
    - Toolbox/
        - ---
        - multi_Cluster.py
        - multi_validity_checker.py
        - ---
    - DBSCAN_digits.py
    - DBSCAN_Iris.py
    - digits with HOG extraction.py
    - hog_copy.py
    - k_means_iris_benchmark.py
    - k-means_digits_benchmark.py
    - k-means_iris-basic.py
    - NearestNeighbourDistances.py
    ```
    helper function for parameter estimation
    ```
    - Playing With GMM and EM.py
    - plot_pca_iris.py

    - data/
     - iris.csv

     

