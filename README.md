# S3-CIMA
Supervised Spatial Single-Cell Image Analysis for identification of disease associated cell type composition in the tissue microenvironment

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S2666389923001988-fx1_lrg.jpg)

S3-CIMA implements a weakly supervised CNN model to identify cell subsets whose frequency distinguishes the considered phenotype labels (i.e., disease associated conditions). The model is adopted from the CellCNN model (Arvaniti and Claassen, 2017), comprising a single layer CNN, a pooling layer and a classification or regression output, and using groups of cell expression profiles (multi-cell inputs) as input. 

# Installation 
S3-CIMA is available on PyPI and can be installed using the command: 

```
pip install s3cima
```

If this does not work, you can clone the repo : 

```
git clone https://github.com/claassenlab/S3-CIMA.git
```

and run the functions in a conda environment with the following packages : 

```
conda create --name sc3cima 
conda activate s3cima
conda install python=3.11 numpy pandas scipy pytorch scikit-learn tqdm matplotlib plotly
```

# Usage
Examples are provided in cima_example.ipynb. Further guidance and documentation to be added soon. 

## run_scima log file 
The model training parameters and outputs is written in a log file including:

•	Important parameters such as K, ncell and anchor celltype

•	Balanced accuracy score on the train/validation/test set 


## S3-CIMA graphical output:

For one anchor population, S3-CIMA generates the following structure of plots: 

```
plots
  |----- test
  |        |----- filter_x
  |        |          |----- all_selected_cells.csv
  |        |          |----- cell_type_proportions.png
  |        |          |----- spatial_plot_sample_xxxx.png
  |        |          |----- spatial_plot_sample_xxxy.png
  |        |          |----- ... 
  |        |
  |        |----- filter_y
  |        |          |----- all_selected_cells.csv
  |        |          |----- cell_type_proportions.png
  |        |          |----- spatial_plot_sample_xxxx.png
  |        |          |----- spatial_plot_sample_xxxy.png
  |        |          |----- ... 
  |        |
  |        |----- ...
  |
  |----- train
           |----- filter_x
           |          |----- all_selected_cells.csv
           |          |----- cell_type_proportions.png
           |          |----- spatial_plot_sample_xxxx.png
           |          |----- spatial_plot_sample_xxxy.png
           |          |----- ... 
           |
           |----- filter_y
           |          |----- all_selected_cells.csv
           |          |----- cell_type_proportions.png
           |          |----- spatial_plot_sample_xxxx.png
           |          |----- spatial_plot_sample_xxxy.png
           |          |----- ... 
           |
           |----- ...
```

The train/test directories correspond to the samples used to train the model and the samples kept aside to test the identified filters on an unseen subset the data.
Strong, generalisable filters should identify similar cell types between train/test provided samples are not wildly heterogeneous. 

### all_selected_cells csv 

This csv contains the cell ids of all the cells in the relevant train or test samples that were selected by the filter.

### cell_type proportions.png

This plot represents the overall proportion of each cell type in the selected cells.

### spatial_plot_sample_xxx.png

For each sample that has at least one selected cell, plots the selected cells in the original spatial dimensions. The number of spatial plots varies per filter - 
some filters might select cells across samples from one condition, other filters might select cells from another.
 

 # Citation

If you use S3-CIMA in your research, please cite our paper:

Sepideh Babaei, Jonathan Christ, Vivek Sehra, Ahmad Makky, Mohammed Zidane, Kilian Wistuba-Hamprecht, Christian M. Schürch, Manfred Claassen,
S3-CIMA: Supervised spatial single-cell image analysis for identifying disease-associated cell-type compositions in tissue, Patterns, Volume 4, Issue 9,
2023, 100829, ISSN 2666-3899, [https://doi.org/10.1016/j.patter.2023.100829](https://www.sciencedirect.com/science/article/pii/S2666389923001988).


# License

S3-CIMA is released under the MIT License. See the LICENSE file for more details.