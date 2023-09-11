import training as tr
import testing as te


if __name__ == "__main__":
    
    print()
    ML_Model = "Linear model - Linear Regression"
    print(f"Used Model: {ML_Model}")
    print()
    tr.Training_The_Model()
    te.LoadAndPredict()
