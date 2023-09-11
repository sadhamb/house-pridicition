# Loading the model:
import joblib

# Loading the test dataset:
from training import Training_The_Model
from training import X_test,y_test
# Testing the Model:
# TrainingTheModel()
from sklearn.metrics import mean_squared_error
def LoadAndPredict():

    # Loading the Model:
    model = joblib.load(filename="Linear_model.sav")


    # Model prediction:
    y_act = model.predict(X_test)


    # accuracy of the model:
    print(f"Model accuracy :{mean_squared_error(y_act,y_test)}")
    print()

