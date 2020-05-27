import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Spam Messages Classification")
    st.sidebar.title("Spam Messages Classification")
    st.sidebar.markdown("Spam or non-spam ?")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("spam.csv",encoding='latin-1')
        data=data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
        label=LabelEncoder()
        data['v1'] = label.fit_transform(data['v1'])
        return data

    #@st.cache(persist=True,allow_output_mutation=True)
    def split(df):
        f = feature_extraction.text.CountVectorizer(stop_words = 'english')
        X = f.fit_transform(df['v2'].values.astype('U'))
        Y = df['v1']
        x_train, y_train, x_test, y_test = train_test_split(X,Y, test_size= 0.35, random_state=42)
        return x_train, y_train, x_test, y_test

    def plot_metrics(metrics):    
        if 'Confusion_Metrics' in metrics:
            st.subheader('Confusion Metrics')
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()
        if 'ROC Curve' in metrics:
            st.subheader('ROC Curve')
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if 'PR Curve' in metrics:
            st.subheader('PR Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    class_names=['Spam', 'Non Spam']

    st.sidebar.subheader("Choose Classifiers")
    classifiers = st.sidebar.selectbox("Classifier",("Support Vector Machine(SVM)","Logistic Regression","Random Forest"))
    if classifiers == "Support Vector Machine(SVM)":
        st.sidebar.subheader("Model Hyperparameter")
        C = st.sidebar.number_input("C {Regularization parameter}", 100, 2000, step=100, key = 'C')
        kernel = st.sidebar.radio('Kernel',("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("What metrics to plote?" ,("Confusion_Metrics","ROC Curve","PR Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Support Vector Machine")
            model = SVC(C=C)
            model.fit(x_train,y_train)
            acc = model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",acc.round(2))
            st.write("Precision: ",precision_score(y_test, y_pred,labels= class_names).round(2))
            st.write("Recall: ",recall_score(y_test, y_pred,labels= class_names).round(2))
            plot_metrics(metrics)
            
    if classifiers == "Logistic Regression":
        st.sidebar.subheader('Hyperparameters')
        C = st.sidebar.number_input("C {Regularization parameter}", 0.01, 10.0, step=0.01, key = 'C')
        max_iter = st.sidebar.slider("Maximum number of iteration?",100,500,key="max_iter")
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion_Metrics", "ROC Curve", "PR Curve"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter= max_iter)
            model.fit(x_train,y_train)
            acc = model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",acc.round(2))
            st.write("Precision: ",precision_score(y_test, y_pred,labels= class_names).round(2))
            st.write("Recall: ",recall_score(y_test, y_pred,labels= class_names).round(2))
            plot_metrics(metrics)
            
    if classifiers == "Random Forest":
        st.sidebar.subheader('Hyperparameters')
        n_estimators = st.sidebar.number_input("The number of trees in the forest?", 100, 5000,step=10,key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree?", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap Sample",("True","False"),key="bootstrap")
        criterion=st.sidebar.radio("Criterion",("gini","entropy"),key="criterion")
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion_Metrics", "ROC Curve", "PR Curve"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators , max_depth =max_depth ,
                                 criterion=criterion,bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            acc = model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",acc.round(2))
            st.write("Precision: ",precision_score(y_test, y_pred,labels= class_names).round(2))
            st.write("Recall: ",recall_score(y_test, y_pred,labels= class_names).round(2))
            plot_metrics(metrics)            
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Spam Classification")
        st.write(df)
    
    
if __name__== "__main__":
    main()
