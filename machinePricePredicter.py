import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import os
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import PolynomialFeatures

def min_max_mean(dataset_df):
	sale_price = dataset_df["SalePrice"].values
	year_made = dataset_df["YearMade"].values
	machine_hours_current_meter = dataset_df["MachineHoursCurrentMeter"].values
	annotation = ["max: ", "min: ", "mean: "]

	sale_price_values = [sale_price.max(), sale_price.min(), sale_price.mean()]
	plt.figure(1)
	plt.title('Mean, max and min sale price')
	plt.scatter([1,1,1], sale_price_values)
	plt.xticks([1],["SalePrice"])
	for i, txt in enumerate(annotation):
		plt.annotate(txt + str(round(sale_price_values[i],0)), (1,sale_price_values[i]))

	year_made_values = [year_made.max(), year_made.min(), year_made.mean()]
	plt.figure(2)
	plt.title('Mean, max and min year made')
	plt.scatter([1,1,1], year_made_values)
	plt.xticks([1],['YearMade'])
	for i, txt in enumerate(annotation):
		plt.annotate(txt + str(round(year_made_values[i],0)), (1,year_made_values[i]))

	machine_hours_current_meter_values = [machine_hours_current_meter.max(), machine_hours_current_meter.min(), machine_hours_current_meter.mean()]
	plt.figure(3)
	plt.title('Mean, max and min machine hours')
	plt.scatter([1,1,1], machine_hours_current_meter_values)
	plt.xticks([1],["MachineHoursCurrentMeter"])
	for i, txt in enumerate(annotation):
		plt.annotate(txt + str(round(machine_hours_current_meter_values[i],0)), (1,machine_hours_current_meter_values[i]))
	plt.ylim((-10, 5000))
	plt.show()

def read_format_dataset():
	dataset_df = pd.read_csv(dir_path + '/dataset/TrainAndValid.csv')

	#drop certain attributes since they were deemed unneccessary
	dataset_df = dataset_df.drop(columns = ['MachineID','ModelID','fiModelDesc', 'fiProductClassDesc', 'ProductGroupDesc', 'fiBaseModel', 
		'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor' ] )
	
	#fill in 0  for every NaN value except of auctioneerID which already has a 0 and machineHours
	dataset_df = dataset_df.fillna({"auctioneerID": "-1"})
	dataset_df = dataset_df.fillna({"MachineHoursCurrentMeter": dataset_df["MachineHoursCurrentMeter"].mean(axis=0, skipna=True)})
	dataset_df = dataset_df.fillna(0)

	#one hot encode all categorical variables
	dataset_df = pd.get_dummies(dataset_df, 
		columns = ['datasource', 'auctioneerID', 'UsageBand', 'ProductSize', 'state', 'ProductGroup', 
		'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 
		'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 
		'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 
		'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type',
		'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 
		'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls'], 
		 prefix = ['datasource', 'auctioneerID', 'UsageBand', 'ProductSize', 'state', 'ProductGroup', 
		'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 
		'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 
		'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 
		'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type',
		'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 
		'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls'])
	

	#fixx date variable to integer
	dataset_df['saledate'] = pd.to_datetime(dataset_df['saledate']) 
	dataset_df['saledate'] = dataset_df['saledate'].apply(dt.datetime.toordinal)

	return dataset_df

def get_outliers(dataset, outliers_fraction):
	print("Finding outliers")
	clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
	clf.fit(dataset)
	result = clf.predict(dataset)
	return result

def save_as_hdf(dataset_df, filename):

	dataset_df.to_hdf(filename, key = 'df', mode = 'w')

def save_importances(model, dataset_df):
	feature_weights = model.feature_importances_
	header = dataset_df.drop(columns = ['SalesID', 'SalePrice']).columns.values

	d = {}
	for key, value in zip(header, feature_weights):
		d[key] = [value]
	excel_df = pd.DataFrame(data=d)

	writer = pd.ExcelWriter('attribute_importance.xlsx', engine='xlsxwriter')
	excel_df.to_excel(writer, sheet_name='Sheet1')
	writer.save()

def grid_search(X_train, y_train):

	print("Making variables")
	n_estimators = [1, 2, 10]#, 50, 100]
	min_samples_leaf = [1, 2, 4]#, 8, 16]
	parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf}
	r2_scorer = make_scorer(r2_score, greater_is_better=True)
	model = RandomForestRegressor()
	grid_obj = GridSearchCV(model, parameters, r2_scorer, cv=3, n_jobs = -1)

	print("Doing a grid search")
	grid_obj.fit(X_train, y_train)

	scores = grid_obj.cv_results_['mean_test_score'].reshape(len(n_estimators),len(min_samples_leaf))

	plot_heat_map(scores, n_estimators, min_samples_leaf)

def plot_heat_map(scores, n_estimators, min_samples_leaf):
	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
	plt.xlabel('n_estimators')
	plt.ylabel('min_samples_leaf')
	plt.colorbar()
	plt.xticks(np.arange(len(n_estimators)), n_estimators)
	plt.yticks(np.arange(len(min_samples_leaf)), min_samples_leaf)
	plt.title('Grid Search R2 score')
	plt.show()

def single_random_forest_test(X_train, y_train, X_test, y_test, dataset_df, n_estimators, min_samples_leaf):

	model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
	print("Train classifier")
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	print("R2 Score: " + str(r2_score(y_test, predictions)))
	print("Mean absolute error: " + str(mean_absolute_error(y_test, predictions)))
	print("Median absolute error: " + str(median_absolute_error(y_test, predictions)))

	save_importances(model, dataset_df)


dir_path = os.path.dirname(os.path.realpath(__file__))

print("Read dataset")
#dataset_df = read_format_dataset()
#save_as_hdf(dataset_df, "dataset_machine_hours.h5")
dataset_df = pd.read_hdf('dataset_machine_hours.h5')

#min_max_mean(dataset_df)

print("Formatting dataset")
X = dataset_df.drop(columns = ['SalesID', 'SalePrice']).values
y = dataset_df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#grid_search(X_train, y_train)
single_random_forest_test(X_train, y_train, X_test, y_test, dataset_df, 10, 4)


