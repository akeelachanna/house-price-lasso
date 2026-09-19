# House Price Prediction Using Lasso Regression

Predicting California house prices using Lasso Regression with automatic feature selection.

## Dataset
- California Housing dataset from scikit-learn
- 20,640 samples with 8 features (median income, house age, rooms, population, etc.)
- Target: Median house value

## Project Workflow
1. Loaded and explored the dataset (EDA with statistics and correlations)
2. Visualized target distribution and feature correlations
3. Standardized features (important for Lasso)
4. Trained Lasso Regression with multiple alpha values
5. Compared Lasso with Linear Regression and Ridge Regression
6. Visualized Lasso feature coefficients (which were kept vs eliminated)

## Key Insights
- Lasso performed automatic feature selection, eliminating less important features
- Model achieved strong R² score while maintaining interpretability
- Feature coefficients visualized to identify key drivers of house prices

## Tools Used
- Python (pandas, numpy, scikit-learn)
- Matplotlib, Seaborn
- Jupyter Notebook

## Files
- `house-price-lasso.ipynb` — Complete notebook with code and outputs
- `lasso_coefficients.csv` — Feature coefficients from the Lasso model
