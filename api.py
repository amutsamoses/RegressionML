import sys
from flask import Flask, request, jsonify
import pandas as pd
import traceback
import joblib



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = lr.predict(query).tolist()
            
            return jsonify({'prediction': str(prediction)})
        
        except:
            return jsonify({'trace': traceback.format_exc()})
        
    else:
        print('Train the model first')
        print('No model here in use')
        
if __name__ == '__main__':
    try:
        # this is for the command line input
        port = int(sys.argv[1])
    except:
        port = 12345 # if there is no any port provided it will be set to 12345
    
    lr = joblib.load('model.pkl')
    print('MODEL LOADED')
    model_columns = joblib.load('model_columns.pkl')
    print('MODEL columns LOADED')
    
    app.run(port=port, debug=True)