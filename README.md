# RT_PyGeo

This study focuses on the exploration and development of a Graph Convolutional Neural Network (GCNN) model
for accurately predicting Collision Cross-Section (CCS) and Retention Time (RT) values of peptides. The datasets
used in this study are derived from the works conducted by Meier et al. and Rosenberger et al. The model
architecture is optimized through hyperparameter tuning, including the selection of the number of hidden layers
and their widths. Evaluation metrics, such as R-squared (R2), mean absolute error (MAE), root mean squared
error (RMSE), and Pearson's correlation coefficient (R), are employed to assess the model's performance. The
model achieves high R2 values of 0.9125 and 0.9414 for CCS and RT, respectively, demonstrating its accuracy
in predicting peptide properties. The study concludes with suggestions for future work, such as incorporating
more amino acid modifications, designing a hierarchical graph structure, and predicting additional peptide
properties. These enhancements would further improve the versatility and comprehensiveness of the peptide
property prediction model.


# Dataset 
1) Rosenberger, G.; Koh, C. C.; Guo, T.; Röst, H. L.; Kouvonen, P.; Collins, B. C.; Heusel, M.; Liu, Y.; Caron, E.; Vichalkovski, A.; Faini, M.; Schubert, O. T.;
Faridi, P.; Ebhardt, H. A.; Matondo, M.; Lam, H.; Bader, S. L.; Campbell, D. S.; Deutsch, E. W.; Moritz, R. L.; Tate, S.; Aebersold, R. A Repository of Assays to
Quantify 10,000 Human Proteins by SWATH-MS. Sci Data 2014, 1 (1), 140031. https://doi.org/10.1038/sdata.2014.31.
2) Ma, C.; Ren, Y.; Yang, J.; Ren, Z.; Yang, H.; Liu, S. Improved Peptide Retention Time Prediction in Liquid Chromatography through Deep Learning. Anal.
Chem. 2018, 90 (18), 10881–10888. https://doi.org/10.1021/acs.analchem.8b02386.

# Acknowledgments
I would like to express my gratitude to [Dr. Camille Bilodeau](https://github.com/cbilodeau2) for her excellent mentorship and insightful
discussions, which have greatly enriched the learning experience throughout this study. I would also like to extend
my appreciation to all the members of the [Bilodeau Group](https://bilodeau-group.com/) for their contributions and participation in the
advancement of this work through our regular meetings. Their support and collaboration have been invaluable in
the progress and success of this work.
