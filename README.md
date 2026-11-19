# What Drives the Price of a Car?

Practical Application 2 uses advertised used-car prices to help dealership appraisal teams find comparable vehicles and check proposed asking prices.

- [Dealer findings and appraisal pilot](dealer_findings.pdf)
- [Executed Jupyter notebook](used_car_price_analysis.ipynb)

## Findings

- **Model year and mileage were the strongest price signals**, followed by manufacturer, cylinder count, and model. Start a comparable search with these fields and verify title and equipment before pricing.
- **Gradient boosting missed prices by $3,249 on average**, a 71% reduction relative to the median-price baseline. Its held-out RMSE was $5,330 and R² was 0.864.
- **Premium listings warrant a closer review.** Above $50,000, the average miss was $11,514 and estimates were $9,435 below asking price on average. Check local comparables and inspection findings before setting an asking price.
- **Pilot the benchmark with recent store activity.** Use actual sales, reconditioning costs, gross margins, and time to sale to evaluate inventory decisions.

## Analysis

The notebook follows all six CRISP-DM stages. It audits 426,880 course-supplied Kaggle listings and retains 322,274 after price and model-year screening and duplicate removal. Training uses 257,202 listings across three group-disjoint validation folds, with a separate 65,072-listing holdout. VIN serves as a grouping key and is excluded from predictors.

| Model | Why it was included | Holdout MAE |
|---|---|---:|
| Median baseline | Measures the error from quoting one training median price for every vehicle. | $11,293 |
| Ordinary least squares | Tests a simple additive relationship between vehicle attributes and price. | $4,796 |
| Ridge regression | Tests whether shrinking linear coefficients improves accuracy and provides adjusted dollar comparisons. | $4,794 |
| Histogram gradient boosting | Captures nonlinear relationships and interactions. It had the lowest cross-validation error. | $3,249 |

Grid search compared three Ridge penalties and four boosting configurations. The winning model used **300 boosting iterations and up to 63 leaves per tree**. The notebook documents preprocessing, the full tuning grid, holdout metrics, price-band errors, permutation importance, and Ridge coefficients.

## Visual evidence

The notebook charts asking-price distribution, mileage, model year, body type, and title status. It also shows which fields the winning model relied on, predicted versus observed prices, and the size and direction of errors by price band. The dealer report includes six plots that support comparable selection and appraisal review.

## Reproduce

1. Use Python 3.14 and install the pinned dependencies with `python -m pip install -r requirements.txt`.
2. Place the course starter dataset at `data/vehicles.csv`. The CSV is excluded from Git.
3. Open the notebook in a Jupyter-compatible editor, select that Python environment, and run all cells from the project directory.

The saved notebook includes the computed results and charts.
