from matplotlib.pyplot import clf
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

#logging  configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error while loading parameters from %s: %s', params_path, e)
        raise
    except Exception as e:
        logger.error('Some error occurred while loading parameters from %s: %s', params_path, e)
        raise

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded successfully from %s',file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise
    



def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict) -> GradientBoostingClassifier:
    try:
        #Define and train the GradientBoosting model
        clf=GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
        clf.fit(X_train,y_train)
        logger.debug('Model trained succesfully')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise
    
    
def main():
    try:
        params=load_params("params.yaml")['train_model']
        #fetch the data from data/processed
        train_data=pd.read_csv('./data/features/train_bow.csv')
        X_train=train_data.iloc[:,0:-1].values
        y_train=train_data.iloc[:,-1].values
        clf=train_model(X_train,y_train,params)
        save_model(clf,'models/model.pkl')
    except Exception as e:
        logger.error('Model building failed: %s', e)
        print(f"Error: {e}")
        


def save_model(model,file_path:str) -> None:
    try:
        #save
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved successfully at %s',file_path)
    except Exception as e:
        logger.error('Error saving model to %s: %s',file_path,e)
        raise
    
if __name__=="__main__":
    main()

