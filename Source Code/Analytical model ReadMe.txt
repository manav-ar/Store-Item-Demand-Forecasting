The analytical model implements a model class that contains the whole pipeline of preprocessing to generating predictions. 

Function definitions

preprocessing: parse the raw date into its constituents.

fit_annual_growthFactor: create a quadratic polynomial model of the yearly sales growth.

create_base_model: create aggregated pivot tables for store/item, day, and month.

predict_annual_growthFactor: return annual growth for 2018.

predict: find the predicted sales on the aggregated tables using the store, item, and date variables.

fit: create the model pipleline by internally calling preprocessing() and create_base_mode()

initTestData: retrieves test data and calls the preprocessing function

evaluates: retrieve test data by calling initTestData(), run predict function on all records, and store the value inside the predictions list


Usage

1. create a model object
2. call model.fit(dataset) method
3. call model.evaluateTest() method to generate predictions on test data
4. predictions can be fetched using model. predictions function