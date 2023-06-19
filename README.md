# pretrained embeddings for hierarchical data representation.


HEARbaselines intallation pip: https://github.com/hearbenchmark/hear-baseline


0) extract_embeddings.py: for each dataset and split(train, val, test and unseentest), extracts embeddings from all the 4 pretrained models and saves as npy,
1) data_functions.py: defines dataset class, Normalizes embeddings for each dataset based on training set stats.
2) evaluate_initial_embeddings.py: for each dataset and model, computes silhouette scores and KNN accuracy results for embeddings PCAtransformed to 3, 4, ..^2 , original_dimension model; generates results table in /results/
