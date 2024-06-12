import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##load model
model=pickle.load(open('WaterQualityClassification.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
pca=pickle.load(open('pca.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    
    # Extract feature values and convert to numpy array
    input_data = np.array(list(data.values()))

    # Ensure input data has the correct number of features
    if input_data.shape[0] != 16:
        return jsonify({"error": "Input data must contain exactly 16 features."}), 400

    # Reshape input data to (1, 16)
    input_data = input_data.reshape(1, -1)
    # Preprocess the input data
    new_data = scaler.transform(input_data)
    new_data_pca=pca.transform(new_data)
    output=model.predict(new_data_pca)
    encoded_class=output
    original_class=encoder.inverse_transform(encoded_class)
    Class_Description={'C1S1':"Low salinity and low sodium waters are good for irrigation and can be used with most crops with no restriction on use on most of the soils. ",
                   'C2S1':"Medium salinity and low sodium waters are good for irrigation and can be used on all most all soils with little danger of development of harmful levels of exchangeable sodium if a moderate amount of leaching occurs. Crops can be grown without any special consideration for salinity control. ",
                   'C3S1':"The high salinity and low sodium waters require good drainage. Crops with good salt tolerance should be selected.",
                  'C3S2':"The high salinity and medium sodium waters require good drainage and can be used on coarse - textured or organic soils having good permeability. ",
                  'C3S3':"These high salinity and high sodium waters require special soil management, good drainage, high leaching and organic matter additions. Gypsum amendments make feasible the use of these waters. ",
                  'C4S1':"Very high salinity and low sodium waters are not suitable for irrigation unless the soil must be permeable and drainage must be adequate. Irrigation waters must be applied in excess to provide considerable leaching. Salt tolerant crops must be selected. ",
                  'C4S2':"Very high salinity and medium sodium waters are not suitable for irrigation on fine textured soils and low leaching conditions and can be used for irrigation on coarse textured or organic soils having good permeability. ",
                  'C4S3':"Very high salinity and high sodium waters produce harmful levels of exchangeable sodium in most soils and will require special soil management, good drainage, high leaching, and organic matter additions. The Gypsum amendment makes feasible the use of these waters. ",
                       'C4S4':"Very high salinity and very high sodium waters are generally unsuitable for irrigation purposes. These are sodium chloride types of water and can cause sodium hazards. It can be used on coarse-textured soils with very good drainage for very high salt tolerant crops. Gypsum amendments make feasible the use of these waters. "
                  }

    if original_class[0] in Class_Description:
        print(original_class[0]+" "+Class_Description[original_class[0]])
    result = {
        "predicted_class": original_class[0],
        "description": Class_Description.get(original_class[0], "No description available")
    }
    #print(output[0])
    return jsonify(result)
if __name__=="__main__":
    app.run(debug=True)