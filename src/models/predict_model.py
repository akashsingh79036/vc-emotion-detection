import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,roc_auc_score 
import logging

#logging  configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')       
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise
    
def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded successfully from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise
    
def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray) -> dict:
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1]
        
        #calculate evaluation metrics
        precision=precision_score(y_test,y_pred,pos_label='happiness')
        recall=recall_score(y_test,y_pred,pos_label='happiness')  
        auc=roc_auc_score(y_test,y_pred_proba)
        accuracy=accuracy_score(y_test,y_pred)
        
        metrics_dict={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation completed with metrics: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics:dict,file_path:str) -> None:
    try:
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Metrics saved successfully at %s', file_path)
    except Exception as e:
        logger.error('Error saving metrics to %s: %s', file_path, e)
        raise
    


def main():
    try:
        clf=load_model('./models/model.pkl')
        
        test_data=load_data('./data/features/test_bow.csv')

        X_test=test_data.iloc[:,0:-1].values
        y_test=test_data.iloc[:,-1].values
        metrics=evaluate_model(clf,X_test,y_test)
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error('Model evaluation failed: %s', e)
        print(f"Error: {e}")


if __name__=="__main__":
    main()


