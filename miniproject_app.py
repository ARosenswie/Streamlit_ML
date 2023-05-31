import streamlit as st
import seaborn as sns
import pandas as pd
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
plt.rcParams["font.family"] = "serif"
plt.rcParams["pdf.fonttype"] = 42


def load_pickles(model_pickle_path):
    with open(model_pickle_path, 'rb') as model_pickle_opener:
        model = pickle.load(model_pickle_opener)
    return model

def make_predictions(test_data):
    model_pickle_path = './models/predicition_model.pkl'
    model = load_pickles(model_pickle_path)
    if 'age' in test_data.columns:
        test_data = test_data.drop('age', axis=1)
    prediction = model.predict(test_data)
    return prediction


with open('./models/valid_predicition_model.pkl','rb') as valid_pred:
    rf_valid_pred = pickle.load(valid_pred)

rf_valid_pred=pd.Series(rf_valid_pred)

df = pd.read_csv('./data/parkinsons_updrs.data.csv')
st.title("Predicting the Age of Future Patients with Parkinson's Disease")

st.header("Within this application, we employ a machine learning algorithm to predict the age of individuals who have a high likelihood of developing Parkinson's Disease.")
url = "https://www.kaggle.com/datasets/thedevastator/unlocking-clues-to-parkinson-s-disease-progressi"
st.write(f"[Click here]({url}) to visit the website.")


fig, ax = plt.subplots()
sns.heatmap(df.corr(), cmap="ocean")
st.write(fig)

st.subheader('We removed sixteen features from the original dataset, which reduces multicollinearity. In addition, we engineer the feature "difference" with the "total_UPDRS" and "motor_UPDRS". ')


df["difference"] = df["total_UPDRS"] - df["motor_UPDRS"]
#Remove features
df = df.drop("index", axis=1)
df = df.drop("subject#", axis=1)
df = df.drop("test_time", axis=1)
df = df.drop("total_UPDRS", axis=1)
df = df.drop("sex", axis=1)
df = df.drop("HNR", axis=1)
df = df.drop("Jitter:RAP", axis=1)
df = df.drop("Jitter(%)", axis=1)
df = df.drop("Jitter:PPQ5", axis=1)
df = df.drop("Jitter:DDP", axis=1)
df = df.drop("Shimmer:APQ3", axis=1)
df = df.drop("Shimmer:APQ5", axis=1)
df = df.drop("Shimmer(dB)", axis=1)
df = df.drop("Shimmer:DDA", axis=1)
df = df.drop("Shimmer", axis=1)
df = df.drop("NHR", axis=1)

Y = df["age"]
X = df.drop("age", axis=1)

cor_matrix = X.corr()
fig2, ax2 = plt.subplots()
sns.heatmap(cor_matrix, annot=False, cmap="ocean");
st.pyplot(fig2)

st.subheader('Seperate the variable that you want to predict via a regression model.  In our case, we desire the age of the patients.')
st.bar_chart(df['age'].value_counts())

train_ratio = 0.70
test_ratio = 0.15
validation_ratio = 0.15

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=validation_ratio/(train_ratio+test_ratio))

st.subheader("To accommodate Scikit-Learn's limitation in creating a validation dataset, we opted to divide the training set into three distinct subsets: 70% for training, 15% for validation, and 15% for testing. This is represented by the following equation:")

st.latex(r'''test = \frac{validation}{train+test}.''')
st.markdown("[Click here for origin of equation.](https://stackabuse.com/scikit-learns-traintestsplit-training-testing-and-validation-sets/)")

st.subheader('Once the training, validation, and testing datasets were generated, we conducted a grid-search against the validation set using five different models: Linear Regression, Decision Tree, Random Forest, eXtreme Gradient Boosting (XGB), and Neural Networks. Among these models, Random Forest demonstrated the highest performance on the validation dataset.')

st.image('./images/table_converter.pdf.png')

st.subheader('Our objective is to visualize the comparison between the predicted values from the XGB model, and the corresponding actual values from the validation dataset.')

fig10, ax10 = plt.subplots()

# Plot the first bar chart on the axes
ax10.hist(df['age'], color='blue', label='Training', bins=100)

# Plot the second bar chart on the same axes
ax10.hist(rf_valid_pred, color ='red', label='Predicted', bins=100)
ax10.set_xlabel('Age')
ax10.set_ylabel('Count')
ax10.legend(loc='best', frameon=False)
st.pyplot(fig10)

st.subheader('We now employ the pre-trained XGB regression model to generate predictions on the testing data. Feel free to use the sliders below to create your own custom predictions.')


if __name__ == '__main__':
    data = pd.read_csv('./data/test.csv')

    # st.text('Select Customer')
    motor_UPDRS = st.slider("Select the patient's motor_UPDR score:",
                           min_value=5, max_value=40, value=5)
    Jitter_Abs = st.slider("Select the patient's Absoulte Jitter:", min_value=0.0, max_value=0.000446, value=0.000001, step=0.00001, format="%.6f")
    Shimmer = st.slider("Select the patient's Shimmer (Variation in vocal range amplitude):", min_value=0.002490, max_value=0.275, value=0.00001, step=0.00001)
    RPDE = st.slider("Select the patient's RPDE (Recurrence Period Density Entropy):", min_value=0.151020	, max_value=0.966080, value=0.00001, step=0.00001)
    DFA = st.slider("Select the patient's DFA (Detrended Fluctuation Analysis):", min_value=0.514040	, max_value=0.865600, value=0.00001, step=0.00001)
    PPE = st.slider("Select the patient's PPE (Pitch Period Entropy):", min_value=0.021983	, max_value=0.731730, value=0.00001, step=0.00001)
    difference = st.slider("Select the patient's difference in motor skills:", min_value=0.025100	, max_value=20.0, value=0.00001, step=0.00001)

    input_dict = {'motor_UPDRS': motor_UPDRS,
                  'Jitter(Abs)': Jitter_Abs,
                  'Shimmer:APQ11': Shimmer,
                  'RPDE': RPDE,
                  'DFA': DFA,
                  'PPE': PPE,
                  'difference': difference
                  }

    input_data = pd.DataFrame([input_dict])
    if st.button("Predict Patient's Age"):
        prediction = make_predictions(input_data)[0].astype(int)
        st.text(f"Predicted Patient's Age: {round(prediction,0)}")