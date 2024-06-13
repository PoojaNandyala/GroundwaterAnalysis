import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
#load model
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
    Class_Description={'C1S1':"""Low salinity and low sodium waters are good for irrigation and can be used with most crops with no restriction on use on most of the soils.
                       -Industry Setup
                        -General Agriculture: Suitable for a wide range of crops with no restrictions.
            -Horticulture: Growing a variety of fruits, vegetables, and ornamental plants.
            -Floriculture: Cultivation of flowers and ornamental plants.
           -Dairy and Livestock Farming: Ensures good quality feed and water for animals.
            -Food Processing: Ensures high-quality raw materials for various food products.
                       """,
                   'C2S1':"""Medium salinity and low sodium waters are good for irrigation and can be used on all most all soils with little danger of development of harmful levels of exchangeable sodium if a moderate amount of leaching occurs. Crops can be grown without any special consideration for salinity control.
                    - Industry Setup
                    -General Agriculture: Suitable for most crops with minimal salinity control.
            -Horticulture: Growing diverse crops with moderate leaching.
            -Livestock Farming: Provides good quality feed and water for animals.
            -Aquaculture: Suitable for fish farming with moderate salinity tolerance.
            -Food Processing: Ensures good quality raw materials for processing industries.""",
                  'C3S1': """High salinity and low sodium waters are suitable for
              - Industry Setup
              - Salt-Tolerant Crop Farming: Barley, cotton, sugar beets, and certain vegetables.
              - Aquaculture: Farming salt-tolerant species like tilapia and shrimp.
              - Salt Production: Solar salt production and extraction.
              - Biotechnology: R&D on salt-tolerant crop varieties.""",
                   'C3S2':"""high salinity and medium sodium waters require good drainage and can be used on coarse - textured or organic soils having good permeability. 
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: Crops that thrive in saline conditions with good drainage.
            -Aquaculture: Salt-tolerant species in well-drained soils.
            -Salt Harvesting: Extraction and processing of salt.",
            -Textile and Dyeing: Industries utilizing saline water, with proper waste management.
                  """,
                  'C3S3':"""high salinity and high sodium waters require special soil management,good drainage, high leaching and organic matter additions Gypsum amendments make feasible the use of these waters.
                  -Industry Setup
                  -Salt-Tolerant Crop Farming: Requires gypsum amendments for soil management.
            -Biotechnology: Developing salt and sodium-tolerant crops.
            -Salt Harvesting: With special soil management practices.
            -Renewable Energy: Solar salt production with high leaching requirements.
                  """,
                  'C4S1':"""Very high salinity and low sodium waters are not suitable for irrigation unless the soil must be permeable and drainage must be adequate. Irrigation waters must be applied in excess to provide considerable leaching. Salt tolerant crops must be selected.
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: Requires permeable soil and high leaching.
            -Aquaculture: Suitable for very salt-tolerant species.
            -Salt Production: With excess irrigation for leaching.
            -Biotechnology: R&D for high salt tolerance in crops.""",
                  'C4S2':"""Very high salinity and medium sodium waters are not suitable for irrigation on fine textured soils and low leaching conditions and can be used for irrigation on coarse textured or organic soils having good permeability.
                  -Industry Setup
                              -Salt-Tolerant Crop Farming: On coarse-textured soils with good permeability.
            -Aquaculture: On suitable soils with high permeability.
            -Salt Harvesting: Requires good drainage and high leaching.
            -Renewable Energy: Solar salt production on permeable soils.""",
                  'C4S3':"""Very high salinity and high sodium waters produce harmful levels of exchangeable sodium in most soils and will require special soil management, good drainage, high leaching, and organic matter additions. The Gypsum amendment makes feasible the use of these waters.
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: With gypsum amendments and special soil management.
            -Biotechnology: Developing crops with high tolerance to salinity and sodium.
            -Salt Production: Requires extensive soil management.
            -Renewable Energy: Solar salt production with high leaching requirements.""",
                       'C4S4':"""Very high salinity and very high sodium waters are generally unsuitable for irrigation purposes. These are sodium chloride types of water and can cause sodium hazards. It can be used on coarse-textured soils with very good drainage for very high salt tolerant crops. Gypsum amendments make feasible the use of these waters.
                       -Industry Setup
                        -Salt-Tolerant Crop Farming: Only for very high salt-tolerant crops with gypsum amendments.
            -Salt Production: On coarse-textured soils with very good drainage.
            -Renewable Energy: Solar salt production under strict soil management."""
                  }

    if original_class[0] in Class_Description:
        print(original_class[0]+" "+Class_Description[original_class[0]])
    result = {
        "predicted_class": original_class[0],
        "description": Class_Description.get(original_class[0], "No description available")
    }
    #print(output[0])
    return jsonify(result)
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    input_data = np.array(data)
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
    Class_Description={'C1S1':"""Low salinity and low sodium waters are good for irrigation and can be used with most crops with no restriction on use on most of the soils.
                       -Industry Setup
                        -General Agriculture: Suitable for a wide range of crops with no restrictions.
            -Horticulture: Growing a variety of fruits, vegetables, and ornamental plants.
            -Floriculture: Cultivation of flowers and ornamental plants.
           -Dairy and Livestock Farming: Ensures good quality feed and water for animals.
            -Food Processing: Ensures high-quality raw materials for various food products.
                       """,
                   'C2S1':"""Medium salinity and low sodium waters are good for irrigation and can be used on all most all soils with little danger of development of harmful levels of exchangeable sodium if a moderate amount of leaching occurs. Crops can be grown without any special consideration for salinity control.
                    - Industry Setup
                    -General Agriculture: Suitable for most crops with minimal salinity control.
            -Horticulture: Growing diverse crops with moderate leaching.
            -Livestock Farming: Provides good quality feed and water for animals.
            -Aquaculture: Suitable for fish farming with moderate salinity tolerance.
            -Food Processing: Ensures good quality raw materials for processing industries.""",
                  'C3S1': """High salinity and low sodium waters are suitable for:
              - Industry Setup
              - Salt-Tolerant Crop Farming: Barley, cotton, sugar beets, and certain vegetables.
              - Aquaculture: Farming salt-tolerant species like tilapia and shrimp.
              - Salt Production: Solar salt production and extraction.
              - Biotechnology: R&D on salt-tolerant crop varieties.""",
                   'C3S2':"""high salinity and medium sodium waters require good drainage and can be used on coarse - textured or organic soils having good permeability. 
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: Crops that thrive in saline conditions with good drainage.
            -Aquaculture: Salt-tolerant species in well-drained soils.
            -Salt Harvesting: Extraction and processing of salt.",
            -Textile and Dyeing: Industries utilizing saline water, with proper waste management.
                  """,
                  'C3S3':"""high salinity and high sodium waters require special soil management,good drainage, high leaching and organic matter additions Gypsum amendments make feasible the use of these waters.
                  -Industry Setup
                  -Salt-Tolerant Crop Farming: Requires gypsum amendments for soil management.
            -Biotechnology: Developing salt and sodium-tolerant crops.
            -Salt Harvesting: With special soil management practices.
            -Renewable Energy: Solar salt production with high leaching requirements.
                  """,
                  'C4S1':"""Very high salinity and low sodium waters are not suitable for irrigation unless the soil must be permeable and drainage must be adequate. Irrigation waters must be applied in excess to provide considerable leaching. Salt tolerant crops must be selected.
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: Requires permeable soil and high leaching.
            -Aquaculture: Suitable for very salt-tolerant species.
            -Salt Production: With excess irrigation for leaching.
            -Biotechnology: R&D for high salt tolerance in crops.""",
                  'C4S2':"""Very high salinity and medium sodium waters are not suitable for irrigation on fine textured soils and low leaching conditions and can be used for irrigation on coarse textured or organic soils having good permeability.
                  -Industry Setup
                              -Salt-Tolerant Crop Farming: On coarse-textured soils with good permeability.
            -Aquaculture: On suitable soils with high permeability.
            -Salt Harvesting: Requires good drainage and high leaching.
            -Renewable Energy: Solar salt production on permeable soils.""",
                  'C4S3':"""Very high salinity and high sodium waters produce harmful levels of exchangeable sodium in most soils and will require special soil management, good drainage, high leaching, and organic matter additions. The Gypsum amendment makes feasible the use of these waters.
                  -Industry Setup
                   -Salt-Tolerant Crop Farming: With gypsum amendments and special soil management.
            -Biotechnology: Developing crops with high tolerance to salinity and sodium.
            -Salt Production: Requires extensive soil management.
            -Renewable Energy: Solar salt production with high leaching requirements.""",
                       'C4S4':"""Very high salinity and very high sodium waters are generally unsuitable for irrigation purposes. These are sodium chloride types of water and can cause sodium hazards. It can be used on coarse-textured soils with very good drainage for very high salt tolerant crops. Gypsum amendments make feasible the use of these waters.
                       -Industry Setup
                        -Salt-Tolerant Crop Farming: Only for very high salt-tolerant crops with gypsum amendments.
            -Salt Production: On coarse-textured soils with very good drainage.
            -Renewable Energy: Solar salt production under strict soil management."""
                  }

    if original_class[0] in Class_Description:
        print(original_class[0]+" "+Class_Description[original_class[0]])
    result = {
        "predicted_class": original_class[0],
        "description": Class_Description.get(original_class[0], "No description available")
    }
    return render_template("home.html", prediction_text="The Groundwater has {}".format(result["description"]))

if __name__=="__main__":
    app.run(debug=True)