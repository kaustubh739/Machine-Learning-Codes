from pathlib import Path
import joblib # harddisk varti model preserved hoil
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# Configuration details
ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "iris_pipeline.joblib"
RANDOM_STATE = 42 # global vari
TEST_SIZE = 0.2


def main():
    Labels = ['Setosa','Versicolor','Virginica']
    pipe = joblib.load(MODEL_PATH)

    Sample = np.array([[5.1,3.5,1.4,0.2]])

    Y_pred = pipe.predict(Sample)[0]

    print("predicted result is : ",Labels[Y_pred])
        
if __name__ == "__main__":
    main()