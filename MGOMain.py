import numpy as np
import math
import nltk
import re
import string
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    cleanr = re.compile('<.*?>')
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(cleanr, ' ', text)        #Removing HTML tags
    text = re.sub(r'[?|!|\'|"|#]',r'',text)
    text = re.sub(r'[.|,|)|(|\|/]',r' ',text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)


    return processed_text


def readData(fileName):
    dataset = pd.read_csv(fileName)
    dataset['tweet'] = dataset['tweet'].apply(preprocess_text)
    #dataset['Sentiment'] = dataset['Sentiment'].map({'negative': 0 , 'positive': 1})
    x = dataset['tweet']
    y = dataset['label']

    #Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state =1
    )
        # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    X_train_tfidf = X_train_tfidf.toarray()
    X_test_tfidf = X_test_tfidf.toarray()

    print(type(X_train_tfidf), type(y_train), X_train_tfidf.shape, y_train.shape)
    print(type(X_test_tfidf), type(y_test), X_test_tfidf.shape, y_test.shape)


    return X_train_tfidf, X_test_tfidf, y_train, y_test

def initialization(SearchAgents_no, dim, ub, lb):
    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    return Positions

def objective_fcn(ghazal, X_train, X_valid, y_train, y_valid):
    total_features = ghazal.shape[0]

    x = np.where(ghazal >= 0.5)
    selected_feature = np.asarray(x)

    num_sel_fe = selected_feature.shape[1]
    if num_sel_fe >= 2:
        X_train_new = X_train[:, selected_feature[0, :]]
        X_valid_new = X_valid[:, selected_feature[0, :]]


        dt_clf = DecisionTreeClassifier(random_state=0)
        dt_clf.fit(X_train_new, y_train)

        out_valid = dt_clf.predict(X_valid_new)
        valid_accuracy = accuracy_score(out_valid, y_valid)

        feature_decrease_rate = num_sel_fe / total_features
        w1 = 0.9
        w2 = 0.1
        #objective_value = w1 * (1 - valid_accuracy) + w2 * feature_decrease_rate
        objective_value = 1 - valid_accuracy
    else:
        objective_value = float("inf")

    return objective_value

def Coefficient_Vector(dim,Iter,MaxIter):

    a2 = -1 + Iter * ((-1)/MaxIter)
    u = np.random.normal(0.5, 0.1, dim)
    v = np.random.normal(0.5, 0.1, dim)

    cofi = np.zeros((4,dim))

    cofi[0,:] =np.random.normal(0.5, 0.1, dim)
    cofi[1,:] = (a2+1) + np.random.rand()
    cofi[2,:] = a2 * np.random.normal(0.5, 0.1, dim)
    cofi[3,:] = np.multiply(np.multiply(u , v ** 2) , np.cos( (np.random.rand() * 2) * u))


    return cofi

def Solution_Imp(X,BestX,lb,ub,N,cofi,M,A,D,Agent, dim):

    NewX = np.zeros((4,dim))

    NewX[0,:]= (ub - lb) * np.random.rand() + lb
    NewX[1,:]= BestX - np.multiply (abs((np.random.randint(2)+1) * M - (np.random.randint(2)+1) * np.multiply(X[Agent,:], A) ), cofi[np.random.randint(4),:])
    NewX[2,:]=(M + cofi[np.random.randint(4),:])+ np.multiply(( (np.random.randint(2)+1) * BestX - (np.random.randint(2)+1)  * X[np.random.randint(N),:] )  , cofi[np.random.randint(4),:]);
    NewX[3,:]= np.multiply( ( X[Agent,:] - D) + ((np.random.randint(2)+1) * BestX - (np.random.randint(2)+1) * M) , cofi[np.random.randint(4),:]);

    return NewX

def  Boundary_Check(NewX, lb, ub):
    # print('---------', NewX)
    for j in range (4):
        FU = np.array(NewX[j,:] > ub)
        FL = np.array(NewX[j,:] < lb)
        T1 = np.invert(FU | FL)
        NewX[j,:]= np.multiply(NewX[j,:] , T1) + np.multiply(ub, FU) + np.multiply(lb , FL)
        # print(New)
    return  NewX

def feature_selection(Ghazalle, threshold):
    x = np.where(Ghazalle >=threshold)
    selected_feature = np.asarray(x)
    return selected_feature

def updated_data(X_train, X_test, selected_feature):
    X_train_new = np.squeeze(X_train[:,selected_feature])
    X_test_new = np.squeeze(X_test[:,selected_feature])

    return X_train_new, X_test_new

def MGO(N, MaxIter, LB, UB, dim, data, label):
    lb=np.ones(dim)*LB;
    ub=np.ones(dim)*UB;

    x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size = 0.3)


    X = initialization(N,dim,ub,lb)
    # print(X)
    Best_pos = np.zeros(dim)
    Best_score = float("inf") # -flaot("inf")

    Sol_Cost = list()
    cnvg = list()
    for Agent in range(N):

        # Calculate the fitness of the population
        Gazal = X[Agent,:].copy()
        # OBJ_A = objective_fcn_test(Gazal)
        OBJ_A = objective_fcn(Gazal, x_train, x_valid, y_train, y_valid )
        Sol_Cost.append(OBJ_A)

        # Update the Best Gazelle if needed
        if OBJ_A < Best_score:
            Best_pos = Gazal.copy()
            Best_score = OBJ_A

    # mainloop
    cnvg.append(Best_score)
    for Iter in range(1, MaxIter):
        for Agent in range(N):

            Perm = np.random.permutation(N)
            RandomSolution = Perm[:min(int(np.ceil(N/3.0)), N)]

            M = np.mean(X[RandomSolution,:] ,axis=0)

            cofi = Coefficient_Vector(dim,Iter,MaxIter)

            A = np.random.normal(0.5, 0.1, dim) *np.exp(2-Iter*(2/MaxIter))
            D = (abs(X[Agent,:]) + abs(Best_pos)) * (2*np.random.rand()-1)


            #  Update the location
            NewX = Solution_Imp(X,Best_pos,lb,ub,N,cofi,M,A,D,Agent, dim)

            # Cost function calculation and Boundary check
            NewX = Boundary_Check(NewX, lb, ub)
            Sol_CostNew = list()
            for j in range(4):
                # Sol_CostNew.append(objective_fcn_test(NewX[j,:]))
                Sol_CostNew.append(objective_fcn(NewX[j,:], x_train, x_valid, y_train, y_valid ))

            # % Adding new gazelles to the herd
            X = np.vstack([X, NewX]) #ok
            Sol_Cost = np.array(Sol_Cost)
            Sol_CostNew = np.array(Sol_CostNew)
            Sol_Cost=np.hstack([Sol_Cost, Sol_CostNew]) #ok

            idbest = np.argmin(Sol_Cost)
            Best_pos=X[idbest,:]


        # Update herd
        SortOrder = np.argsort(Sol_Cost)
        Sol_Cost = np.sort(Sol_Cost)
        X = X[SortOrder,:]

        Best_pos = X[0,:];
        Best_score = Sol_Cost[0]
        X=X[0:N,:];

        Sol_Cost=Sol_Cost[0:N];
        cnvg.append(Best_score)
        BestF=Best_score

        if Iter % 1 == 0:
            print(["At iteration " + str(Iter) + " the best fitness is " + str(Best_score)])

    return Best_score,Best_pos,cnvg


def sentiment_analysis(X_train, X_test, y_train, y_test, selected_feature):

    # Flatten the selected features indices
    selected_feature = selected_feature.flatten()

    X_train_new, X_test_new = updated_data(X_train, X_test, selected_feature)

    # Train a classifier (you can choose a different classifier based on your preference)
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(X_train_new, y_train)

    # Predictions on the test set
    y_pred = classifier.predict(X_test_new)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('Sentiment Analysis Evaluation Metrics:')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

    return accuracy, precision, recall, f1

def main():
    fileName = "dataset.csv"

    x_train, x_test, y_train, y_test = readData(fileName)

    SearchAgents_no = 30  # Number of search agents

    #  Load details and requirements 
    dim = x_train.shape[1] # Number of features
    Max_iteration = 500
    lb = 0
    ub = 1

    Best_score,Best_pos,MGO_cg_curve  = MGO(SearchAgents_no,Max_iteration,lb,ub,dim, x_train, y_train)
    print('=========================================================================================')
    print('Best answer :', Best_pos)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    selected_feature = feature_selection(Best_pos, 0.5)
    print('Selected Feature Indices:', selected_feature)
    print('Number of Selected Features:', selected_feature.shape[1])

    # Use selected features for sentiment analysis
    sentiment_analysis(x_train, x_test, y_train, y_test, selected_feature)

    plt.plot(MGO_cg_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Objective value')
    plt.title('MGO convergence curve')
    plt.show()



if __name__ == "__main__":
    main()