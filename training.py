import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def Training_The_Model():
    # Loading the data.csv dataset to dataframe:
    df = pd.read_csv('data.csv')

    # Droping the unwanted columns:
    df.drop(columns= ['yr_renovated','yr_built','statezip'],inplace = True)

    # Changing the dtypes of columns as categorical dtype:
    df.waterfront = df.waterfront.astype(object)
    df.view = df.view.astype(object)
    df.floors = df.floors.astype(object)
    df.condition = df.condition.astype(object)

    # select categorical dtype columns only:
    df_cat = df.select_dtypes(['object'])

    # Feature engineering:
    le = LabelEncoder()

    # applying to the categorical column:
    df_cat_le = pd.DataFrame()
    for i in df_cat:
        df_cat_le[i]=le.fit_transform(df_cat[i])    
        # print(i)

    # Select numerical dtype columns only:
    df_num = df.select_dtypes(['float64','int64'])

    # Standardisation for the numerical columns:
    sc = StandardScaler()

    # Creating new dataframe for standardised numerical column:
    df_num_sc = pd.DataFrame(sc.fit_transform(df_num),columns = df_num.columns)

    # concatenating the categorical and numerical dataframe:
    df_new = pd.concat([df_cat_le,df_num_sc],axis = 1)
    df_new.head()

    # spliting the Features and Label
    Features = df_new.drop(['price'],axis=1)
    Label = df_new.price

    global X_test,y_test
    # Spliting the data for training and testing section:
    X_train,X_test,y_train,y_test = train_test_split(Features,Label,test_size = 0.3)

    # sorting the data with the index order:
    X_train.sort_index(inplace = True)
    X_test.sort_index(inplace = True)
    y_train.sort_index(inplace = True)
    y_test.sort_index(inplace = True)

    # Implementation of Linear Regression machine learning model algorithm:
    model = LinearRegression()
    model.fit(X_train,y_train)

    # Saving the model 

    file = 'Linear_model.sav'
    joblib.dump(model,file)

Training_The_Model()