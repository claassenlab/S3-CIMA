# S3-CIMA
Supervised Spatial Single-Cell Image Analysis for identification of disease associated cell type composition in the tissue microenvironment

S3-CIMA implements a weakly supervised CNN model to identify cell subsets whose frequency distinguishes the considered phenotype labels (i.e., disease associated conditions). The model is adopted from the CellCNN model (Arvaniti and Claassen, 2017), comprising a single layer CNN, a pooling layer and a classification or regression output, and using groups of cell expression profiles (multi-cell inputs) as input. 

# Usage
Examples are provided in S3CIMA_example.ipynb. Further guidance to be added soon. 

# run_scima log file 
The model training parameters and outputs is written in a log file including:

•	Best model validation accuracy  

•	std of validation accuracies across models

•	multi-cell inputs size

•	Accuracy score on the test set 

•	Anchor

•	nset which is the number of anchor cell in each image.


# plot_results output:

•	clustered_filter_weights.pdf:

Filter weight vectors from all trained networks that pass a validation accuracy threshold, grouped in clusters via hierarchical clustering. Each row corresponds to a filter. The last column(s) indicate the weight(s) connecting each filter to the output class(es). Indices on the y-axis indicate the filter cluster memberships, as a result of the hierarchical clustering procedure. This plot generates from the CellCNN model.

•	consensus_filter_weights.pdf :
One representative filter per cluster is chosen (the filter with minimum distance to all other members of the cluster). This plot generates from the CellCNN model.

•	best_net_weights.pdf :

Filter weight vectors of the network that achieved the highest validation accuracy. This plot generates from the CellCNN model.

•	filter_response_differences.pdf :

Difference in cell filter response between classes for each consensus filter. To compute this difference for a filter, we first choose a filter-specific class, that's the class with highest output weight connection to the filter. Then we compute the average cell filter response (value after the pooling layer) for validation samples belonging to the filter-specific class (v1) and the average cell filter response for validation samples not belonging to the filter-specific class (v0). The difference is computed as v1 - v0. For regression problems, we cannot compute a difference between classes. Instead we compute Kendall's rank correlation coefficient between the predictions of each individual filter (value after the pooling layer) and the true response values. This plot helps decide on a cutoff (filter_diff_thres parameter) for selecting discriminative filters. This plot generates from the CellCNN model.

•	cdf_filter_i.pdf :

Cumulative distribution function of cell filter response for filter i. This plot helps decide on a cutoff (filter_response_thres parameter) for selecting the responding cell population. This plot generates from the CellCNN model.

•	g.txt

A one-column text File indicating the corresponding filter response for each cell of the input data.


