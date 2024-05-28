import streamlit as st
import numpy as np
import pickle


loaded_model = pickle.load(open('SajivGen/Churn-prediction-Classifier-ML-models/blob/main/Classification_models.ipynb/DecisionTree.pkl', 'rb'))


def churn_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshapped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshapped)
    print(prediction)
    if (prediction[0]== 0):
        return 'Customer will not churn'
    else:
        return 'Customer will churn'
    
def main():
    
    st.title('DTH Churn prediction')
    Tenure = st.slider('Tenure',0,105)
    City_Tier = st.selectbox('City_Tier',(1,2,3))
    CC_Contacted_LY = st.slider('CC_Contacted_LY',0,250)
    Payment = st.selectbox('Payment',(1,2,3,4,5))
    Gender	= st.selectbox('Gender Female 1, Gender Male 2',(1,2))
    Service_Score = st.selectbox('Service_Score',(0,1,2,3,4,5))
    Account_user_count = st.selectbox('Account_user_count',(1,2,3,4,5,6))
    Account_segment	= st.selectbox('Account_segment',(1,2,3,4,5))
    CC_Agent_Score	= st.selectbox('CC_Agent_Score',(1,2,3,4,5))   
    Marital_Status	= st.selectbox('Marital_Status Single: 1 Married: 2 Divorced:3',(1,2,3))
    Rev_per_month	= st.slider('Rev_per_month',0,250)
    Complain_ly	= st.selectbox('Complain_ly',(0,1))
    Rev_growth_yoy	= st.slider('Rev_growth_yoy',0,50)
    Coupon_used_for_payment	= st.slider('Coupon_used_for_payment',0,25)
    Day_Since_CC_connect = st.slider('Day_Since_CC_connect',0,100)
    Cashback = 	st.slider('Cashback',0,2500)
    Login_device = st.selectbox('Login_device',(1,2))

    
    makeprediction = ''
    
    if st.button('Churn'):
        makeprediction = churn_prediction([Tenure, City_Tier, CC_Contacted_LY, Payment, Gender, Service_Score,
                                            Account_user_count, Account_segment, CC_Agent_Score, Marital_Status, Rev_per_month, 
                                            Complain_ly, Rev_growth_yoy, Coupon_used_for_payment, Day_Since_CC_connect, Cashback,
                                            Login_device])
                 
    st.success(makeprediction)
    
                 
if __name__ == '__main__':
    main()
