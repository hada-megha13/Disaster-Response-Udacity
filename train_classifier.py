import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import nltk
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("messages_df", con=engine)
    
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    txt=re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tkns=word_tokenize(txt)
    lemma=WordNetLemmatizer()
    prcd_txt=[]
    for t in tkns:
        clean_t=lemma.lemmatize(t)
        prcd_txt.append(clean_t)
    return prcd_txt


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #parameters = {
    #    'clf__estimator__n_estimators': [50, 100, 200],
    #    'clf__estimator__min_samples_split': [2, 3, 4]
    #} 

    #cv = GridSearchCV(pipeline, parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    for r_no, col in enumerate(category_names):
        print(r_no)
        print(col)
        print(classification_report(Y_test[col], y_pred[:, r_no]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()