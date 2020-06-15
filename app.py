from flask import Flask
from flask_restful import Resource,Api
import pandas as pd
import numpy as np
np.random.seed(1234)  
PYTHONHASHSEED = 0
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from flask import jsonify



# load pickle model
filename = 'model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


# load data
test_data = pd.read_csv('test_data.csv') #file

app = Flask(__name__)
api = Api(app)



@app.route('/get-status/<int:input_ID>/', methods=['GET'])
def fn(input_ID):
    print('The inputted Asset:', int(input_ID))
    print(test_data.shape)


    # selecting the data for a particular asset
    if int(input_ID) < 5:
        asset_1 = test_data[test_data['Equip_ID'] == int(input_ID)].tail(8)
    else:
        print("Asset ID does not exist so providing the health of available Assest IDs")
        asset_1 = test_data[test_data['Equip_ID'] == 1].tail(8).append(test_data[test_data['Equip_ID'] == 2].tail(8)).    append(test_data[test_data['Equip_ID'] == 3].tail(8)).append(test_data[test_data['Equip_ID'] == 4].tail(8))



    # data preprocessing

    # dropping the columns which are not required
    asset_1.drop('Machine',inplace=True,axis=1)

    # converting categorical variables into dummy
    asset_1 = pd.get_dummies(asset_1, columns=['Fuel Level', 'Engine', 'PayLoad', 
                                                'Parking Brake', 'MotorTemp'])


    # MinMax normalization
    asset_1['Cycle_Norm'] = asset_1['Cycle']
    cols_normalize = asset_1.columns.difference(['Equip_ID','Cycle'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(asset_1[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=asset_1.index)
    join_df = asset_1[asset_1.columns.difference(cols_normalize)].join(norm_test_df)
    asset_1 = join_df.reindex(columns = asset_1.columns)



    # Columns should be there in dataframe
    neccessary_columns = ['Battery voltage','Coolant Pressure','Cycle_Norm','Engine Coolant Temperature','Engine Oil Pressure',
     'Engine Oil Temperature','Engine_Normal','Engine_Over Speed','Exhaust Temperature','Fuel Level_High','Fuel Level_Low',
     'Fuel Level_Medium','Fuel Temperature','Hydraulic Oil Tank Temperature','MotorTemp_High','MotorTemp_Medium',
     'MotorTemp_Normal','Parking Brake_OFF','Parking Brake_ON','PayLoad_Not OK','PayLoad_OK','Steering Oil Pressure',
         'Wheel Motor Temp']

    # finding and adding columns are not in dataframe from neccessary list
    new_col = list(set(neccessary_columns) - set(list(asset_1.columns)))

    # column count
    col_count = len(new_col)



    if col_count>0:
        asset_1[new_col] = pd.DataFrame([[0] * col_count], index=asset_1.index)
    else:
        pass
        



    # pick a window size of 14 cycles
    sequence_length = 7

    # function to reshape features into (samples, time steps, features) 
    def gen_sequence(id_df, seq_length, seq_cols):
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]
            
            
    # required columns 
    sequence_cols = list(asset_1.columns.difference(['Equip_ID','Cycle']))


    # generator for the sequences
    seq_gen = (list(gen_sequence(asset_1[asset_1['Equip_ID']==id], sequence_length, sequence_cols)) 
               for id in asset_1['Equip_ID'].unique())


    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    #seq_array.shape



  

    dict_ = {}

    # predicting the value
    pred_value = np.squeeze(loaded_model.predict_classes(seq_array))
    if pred_value.size == 1:
        pred_value = np.squeeze(loaded_model.predict_classes(seq_array)).item()
        if pred_value==0:
            dict_[input_ID] = 0
            print("Mantainance not required for Asset ID: ",input_ID)
        elif pred_value==1:
            dict_[input_ID] = 1
            print("Mantainance required for Asset ID: ",input_ID)
        else:
            print("No Result")
        
    elif pred_value.size > 1:
        i = 1
        for val in pred_value:
            if val==0:
                dict_[i] = 0
                print("Mantainance not required for Asset ID: ",i)
            elif val==1:
                dict_[i] = 1
                print("Mantainance required for Asset ID: ",i)
            else:
                print("No Result")
            i = i+1
    else:
        print("No Result")
    
    
    return jsonify(dict_)


 
if __name__ == "__main__":
    app.run(debug=None,threaded=False)