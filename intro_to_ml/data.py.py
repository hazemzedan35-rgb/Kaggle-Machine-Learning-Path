import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

## load and clean Data 
df = pd.read_csv("Student_Performance.csv")
df_data = df.dropna()

## splitting Data into x and y 
y = df_data.overall_score
df_features = ['study_hours', 'attendance_percentage']
X = df_data[df_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

## define and fit 
# define
model = RandomForestRegressor(random_state=1)

# fit 
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
model_mae = mean_absolute_error(val_y, val_predictions)
print(f"mean absolute error is {model_mae}")

#### predict 
new_student = pd.DataFrame({'study_hours': [4], 'attendance_percentage': [99]})
predictions = model.predict(new_student)
print(predictions)