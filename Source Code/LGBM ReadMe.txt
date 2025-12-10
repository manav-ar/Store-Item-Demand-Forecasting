This file is regarding the LGBM model that was implemented for store item forecasting.

For this respective model, we perform precprocessing to handle outliers, remove seasonality and find critical stores. This is done using the following steps:

	1. Finding the slope and intercept of a linear fit for sale values grouped by store, item and day of week, followed by fitting a linear model to the sale values to use this as a trend. 
	
	2. Removal of increasing trend and yearly seasonality from the sale values. Followed by normalisation of the stationary sale values and identification of the outliers.
	
	3. Handling the outliers using interpolation to get corrected sales and building the expanding mean sale values which are grouped by store, item, (day of week, month, and quarter).
	
	4. Finding stores and items whose mean sales value is below the 50% percentile. While predicting for year 2018 of these stores, these items will be multiplied by a factor smaller than one.

Once, preprocessing is done, we move to the actual training of the LGBM model. We made use of grid search to tune the hyperparameters and the values we finally used were as follows:
	
	parameter = value |	Task = Train | Boosting = typeGBDT | objective = Regression | Num leaves = 10 | Max depth = 3 | Metric = SMAPE | Learning rate = 0.1 | Boosting rounds = 10000

Employing this solution gave us the respective results:

	Public Leaderboard = 13.94421 | Private Leaderboard = 12.67741 | Validation Score = 12.59647

This approach although seemed to work well, but doesn't achieve the best results possible. This is primarily because of the fact that the data seems to synthetically generated in nature and thus statistical models seem to have an easier task of forecasting this data.