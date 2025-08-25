# Decision Tree (ID3) Lab Report

**Student Name:** Naman Nagar  
**SRN:** PES2UG23CS361  
**Campus:** EC  
**Section:** F  
**Date:** August 25, 2025  

---

## 1. Objective
The objective of this lab was to implement and evaluate the ID3 Decision Tree algorithm using both NumPy (reference with sklearn) and PyTorch frameworks. The aim was to measure model performance across multiple datasets, analyze the complexity of the constructed trees, and compare outcomes between different frameworks.

---

## 2. Datasets Used
1. **mushrooms.csv** – Classify mushrooms as edible or poisonous.  
2. **tictactoe.csv** – Predict game outcomes (win/loss) based on board positions.  
3. **Nursery.csv** – Predict admission recommendation levels based on family and social attributes.  

---

## 3. Methodology
- Preprocessed all datasets (categorical features encoded as numerical values).  
- Implemented ID3 components:  
  - Entropy calculation  
  - Average information of attributes  
  - Information gain  
  - Best attribute selection  
- Constructed decision trees using both NumPy/PyTorch implementations.  
- Evaluated using Accuracy, Precision, Recall, F1-score (both weighted and macro).  
- Measured structural complexity: depth, number of nodes, leaf nodes, internal nodes.  

---

## 4. Results

### 4.1. Mushrooms Dataset
- **PyTorch**
  - Accuracy: 100%  
  - Weighted Precision/Recall/F1: 1.0000  
  - Macro Precision/Recall/F1: 1.0000  
  - Depth: 4 | Total Nodes: 29 | Leaves: 24 | Internal: 5  

- **sklearn**
  - Accuracy: 100%  
  - Weighted and Macro scores: 1.0000  
  - Depth: 4 | Total Nodes: 29 | Leaves: 24 | Internal: 5  

**Observation:** Perfect classification due to highly discriminative features like odor and spore-print-color. Both frameworks produced identical results.

---

### 4.2. TicTacToe Dataset
- **PyTorch**
  - Accuracy: 87.30%  
  - Weighted F1: 0.8734 | Macro F1: 0.8613  
  - Depth: 7 | Total Nodes: 281 | Leaves: 180 | Internal: 101  

- **sklearn**
  - Accuracy: 88.36%  
  - Weighted F1: 0.8822 | Macro F1: 0.8680  
  - Depth: 7 | Total Nodes: 260 | Leaves: 165 | Internal: 95  

**Observation:** Good accuracy, but lower than mushroom dataset due to overlapping decision boundaries and less separable features. Sklearn produced a slightly smaller tree with higher accuracy.

---

### 4.3. Nursery Dataset
- **PyTorch**
  - Accuracy: 98.67%  
  - Weighted F1: 0.9872 | Macro F1: 0.7628  
  - Depth: 7 | Total Nodes: 952 | Leaves: 680 | Internal: 272  

- **sklearn**
  - Accuracy: 98.87%  
  - Weighted F1: 0.9887 | Macro F1: 0.9576  
  - Depth: 7 | Total Nodes: 983 | Leaves: 703 | Internal: 280  

**Observation:** Both frameworks performed very well. Weighted metrics were near perfect, but macro metrics revealed class imbalance (rare classes like *very_recom* and *spec_prior* were harder to classify). Sklearn achieved slightly better macro scores.

---

## 5. Observations & Analysis
- **Performance:**  
  - Mushroom dataset achieved perfect accuracy because of strong predictive features.  
  - TicTacToe dataset showed moderate accuracy (~87–88%) as decision boundaries are not perfectly separable.  
  - Nursery dataset achieved high weighted accuracy (~99%) but macro metrics showed some classes were underrepresented.  

- **Tree Complexity:**  
  - Simpler datasets (Mushrooms) → shallow trees, fewer nodes.  
  - More complex/multi-class datasets (Nursery) → deeper and larger trees.  
  - TicTacToe produced medium-depth trees with a moderate number of nodes.  

- **Framework Comparison:**  
  - Both PyTorch and sklearn gave consistent results, with sklearn often producing slightly smaller trees and better macro metrics.  
  - Differences are likely due to optimization and implementation details in sklearn’s library.  

- **Overfitting Indicators:**  
  - None observed in Mushrooms (clear separation).  
  - Slight overfitting possible in Nursery due to large tree size.  

---

## 6. Conclusion
- The ID3 algorithm worked effectively on all datasets, with performance strongly dependent on dataset characteristics.  
- Mushroom dataset was perfectly separable, Nursery dataset was high performing but imbalanced, and TicTacToe was moderately accurate.  
- Sklearn implementation was slightly more optimized than custom PyTorch implementation, though results were broadly consistent.  
- Decision trees remain interpretable and useful, especially when key features (like odor in mushrooms) dominate predictions.  
- However, for highly imbalanced or overlapping datasets, performance may be limited and improvements could be made using pruning or ensemble methods (e.g., Random Forests).  

---

## 7. References
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/tree.html  
- PyTorch Documentation: https://pytorch.org/docs/stable/  
- Course Material: Machine Learning Lab – ID3 Decision Tree Assignment  
