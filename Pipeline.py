from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_iris

# Load dataset

iris = load_iris()
X,y = iris.data,iris.target

# Create pipeline : Scaling + Logistic Regression

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('cassifier', LogisticRegression())
])

# Train pipeline
pipeline.fit(X,y)

# Save pipeline to HDD
joblib.dump(pipeline,'iris_pipeline.joblib')
print("pipeline saved as iris_pipeline.joblib")

# Load pipeline
loaded_pipeline = joblib.load('iris_pipeline.joblib')

# predict with pipeline
sample_data = [[5.8,2.8,5.1,2.4]]
prediction = loaded_pipeline.predict(sample_data)

print("predicted class: ", prediction)