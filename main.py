import numpy as np
import pickle
import streamlit as st 


#Loading the saved Model

loaded_model =  pickle.load(open('trained_diabetic_model.sav', 'rb'))

input_data = (6,117,96,0,0,28.7,0.157,30)
def diabetic_prediction(inp):
    inp_array = np.asarray(inp)
    input_data_reshaped = inp_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]== 0):
        return('The Person is Non-Diabetic')
    else:
        return('The Person is Diabetic')
        
def main():
    st.title('Diabetes Prediction Web App')
    
    Pregnancies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
# Code for Prediction 
    diagonosis =''

    if (st.button('Diabetes Result')):
        diagonosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure,SkinThickness, Insulin,
        BMI, DiabetesPedigreeFunction, Age])
        
        st.success(diagonosis)
        
if __name__ == '__main__':
    main()
    