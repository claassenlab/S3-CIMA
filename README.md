# S3-CIMA
Supervised Spatial Single-Cell Image Analysis for identification of disease associated cell type composition in the tissue microenvironment

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S2666389923001988-fx1_lrg.jpg)

S3-CIMA implements a weakly supervised CNN model to identify cell subsets whose frequency distinguishes the considered phenotype labels (i.e., disease associated conditions). The model is adopted from the CellCNN model (Arvaniti and Claassen, 2017), comprising a single layer CNN, a pooling layer and a classification or regression output, and using groups of cell expression profiles (multi-cell inputs) as input. 

# Usage
Examples are provided in cima_example.ipynb. Further guidance and documentation to be added soon. 

# run_scima log file 
The model training parameters and outputs is written in a log file including:

•	Best model validation accuracy  

•	std of validation accuracies across models

•	multi-cell inputs size

•	Balanced accuracy score on the test set 

•	Anchor

•	nset which is the number of anchor cell in each image.


# plot_results output:

Plotting not yet added ! Will be done very soon.


 # Citation

If you use S3-CIMA in your research, please cite our paper:

Sepideh Babaei, Jonathan Christ, Vivek Sehra, Ahmad Makky, Mohammed Zidane, Kilian Wistuba-Hamprecht, Christian M. Schürch, Manfred Claassen,
S3-CIMA: Supervised spatial single-cell image analysis for identifying disease-associated cell-type compositions in tissue, Patterns, Volume 4, Issue 9,
2023, 100829, ISSN 2666-3899, [https://doi.org/10.1016/j.patter.2023.100829](https://www.sciencedirect.com/science/article/pii/S2666389923001988).


# License

S3-CIMA is released under the MIT License. See the LICENSE file for more details.