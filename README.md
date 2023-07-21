# RT_PyGeo

This study successfully developed a model for predicting CCS and RT values of peptides using a Graph
Convolutional Neural Network approach. The architecture of the model, consisting of multiple Graph 
Convolutional Layers and Linear Layers with ReLU activation, was optimized through hyperparameter tuning.
The model demonstrated promising results, achieving high R2 values and accurately predicting peptide properties.
The use of the validation set allowed for the evaluation of the model's generalization ability to unseen data. The
findings suggest that the chosen LR and epoch sizes effectively facilitated convergence and reduced
computational time. The model's performance indicated minimal overfitting, showcasing its ability to generalize
well. Overall, these findings provide valuable insights and a solid starting point for future research in the field of
peptide property prediction


# Dataset 
1) Rosenberger, G.; Koh, C. C.; Guo, T.; Röst, H. L.; Kouvonen, P.; Collins, B. C.; Heusel, M.; Liu, Y.; Caron, E.; Vichalkovski, A.; Faini, M.; Schubert, O. T.;
Faridi, P.; Ebhardt, H. A.; Matondo, M.; Lam, H.; Bader, S. L.; Campbell, D. S.; Deutsch, E. W.; Moritz, R. L.; Tate, S.; Aebersold, R. A Repository of Assays to
Quantify 10,000 Human Proteins by SWATH-MS. Sci Data 2014, 1 (1), 140031. https://doi.org/10.1038/sdata.2014.31.
2) Ma, C.; Ren, Y.; Yang, J.; Ren, Z.; Yang, H.; Liu, S. Improved Peptide Retention Time Prediction in Liquid Chromatography through Deep Learning. Anal.
Chem. 2018, 90 (18), 10881–10888. https://doi.org/10.1021/acs.analchem.8b02386.
