# Transformer with Convolution and Graph-Node co-embedding: A accurate and interpretable vision backbone for predicting gene expressions from local histopathological image

Xiao Xiao1,2, Yan Kong1,2, Zuoheng Wang3, Hui Lu2,1,4,*

1State Key Laboratory of Microbial Metabolism, Joint International Research Laboratory of Metabolic and Developmental Sciences, Department of Bioinformatics and Biostatistics, School of Life Sciences and Biotechnology, Shanghai Jiao Tong University, Shanghai, China

2SJTU-Yale Joint Center for Biostatistics and Data Science, National Center for Translational Medicine, Shanghai Jiao Tong University, Shanghai, China

3Department of Biostatistics, Yale University, New Haven, CT, United States

4Center for Biomedical Informatics, Shanghai Children’s Hospital, Shanghai, China

*** Correspondence: 
** Hui Lu 
 [huilu@sjtu.edu.cn](mailto:huilu@sjtu.edu.cn)

**Keywords: deep learning, breast cancer, convolutional neural network, graph neural network, transformer, spatial transcriptomics.** 

 

**Highlights**

l First deep learning model to integrate CNN, GNN, and transformer for image analysis

l An interpretable model that uses cell morphology and organizations to predict genes

l Higher gene expression prediction accuracy without global information

l Accurately predicted genes are related to immune escape and abnormal metabolism

l Predict important biomarkers for breast cancer accurately from cheaper images

# Abstract

Inferring gene expressions from histopathological images has always been a fascinating but challenging task due to the huge differences between the two modal data. Previous works have used modified DenseNet121 to encode the local images and make gene expression predictions. And later works improved the prediction accuracy of gene expression by incorporating the coordinate information from images and using all spots in the tissue region as input. While these methods were limited in use due to model complexity, large demand on GPU memory, and insufficient encoding of local images, thus the results had low interpretability, relatively low accuracy, and over-smooth prediction of gene expression among neighbor spots. In this paper, we propose TCGN, (Transformer with Convolution and Graph-Node co-embedding method) for gene expression prediction from H&E stained pathological slide images. TCGN consists of convolutional layers, transformer encoders, and graph neural networks, and is the first to integrate these blocks in a general and interpretable computer vision backbone for histopathological image analysis. We trained TCGN and compared its performance with three existing methods on a publicly available spatial transcriptomic dataset. Even in the absence of the coordinates information and neighbor spots, TCGN still outperformed the existing methods by 5% and achieved 10 times higher prediction accuracy than the counterpart model. Besides its higher accuracy, our model is also small enough to be run on a personal computer and does not need complex building graph preprocessing compared to the existing methods. Moreover, TCGN is interpretable in recognizing special cell morphology and cell-cell interactions compared to models using all spots as input that are not interpretable. A more accurate omics information prediction from pathological images not only links genotypes to phenotypes so that we can predict more biomarkers that are expensive to test from histopathological images that are low-cost to obtain, but also provides a theoretical basis for future modeling of multi-modal data. Our results support that TCGN is a useful tool for inferring gene expressions from histopathological images and other potential histopathological image analysis studies.

![image](./data/1.png)

