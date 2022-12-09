# RENGE
### REgulatory Network inference using GEne perturbation data


<!-- <p align="center">
  <img src="https://user-images.githubusercontent.com/20146973/190888505-da05517c-7e06-4864-88c7-8caa26f1844a.jpg" width="300" height="300"/>
</p> -->


RENGE infers gene regulatory networks (GRNs) from time-series single-cell CRISPR analysis data.


## Install
```
pip install renge
```

## Usage
### Network inference
```
reg = Renge()
A = reg.estimate_hyperparams_and_fit(X, E)
A_qval = reg.calc_qval()
```

**input**  
E : C x G pandas DataFrame of expression data. The expression data should be normalized and log-transformed.   
X : C x (G+1) pandas DataFrame. The rows of X[:, :G] are one-hot vectors indicating the knocked out gene in each cell. The last column of X indicates the sampling time of each cell.  

Here,  
C : The number of cells.  
G : The number of genes included in the GRN. 

**output**  
A : G x G pandas DataFrame. (i, j) element of A is a regulatory coefficient from gene j to gene i.   
A_qval : G x G pandas DataFrame. (i, j) element of A_qval is a q-value for (i, j) element of A.  

### Prediction of expression changes after gene knockout
```
reg = Renge()
reg.estimate_hyperparams_and_fit(X_train, E_train)   # train the model
E_pred = reg.predict(X_pred)
```
**input**  
E_train : C x G pandas DataFrame of expression data. The expression data should be normalized and log-transformed.   
X_train : C x (G+1) pandas DataFrame. The rows of X_train[:, :G] are one-hot vectors indicating the knocked out gene in each cell. The last column of X_train indicates the sampling time of each cell.    
X_pred : T x (G+1) pandas DataFrame. The rows of X[:, :G] are real-valued vectors indicating expression change of target gene of perturbation. For knockout or knockdown, values should be negative. The last column of X_pred indicates the time at which expressions are predicted 

Here,  
T : The number of timepoints where expressions are predicted.  

**output**  
E_pred : T x G pandas DataFrame of predicted expression.


## Reference
TBD

