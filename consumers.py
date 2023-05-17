from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import ast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

#Imports used for python tf-idf
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
import sqlalchemy
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
import nltk
nltk.download('words')

app = Flask(__name__)
api = Api(app)

@app.route("/reply", methods=["GET", "POST"])
def LoadData():
    user_input = request.args.get('question')

    def tfidf_custom_scoring(input_text):
        # Fit the data to the vectorizer
        vectorizer.fit(DS1_data['Consumer complaint narrative'])

        # Adds the input from the user to the fit using an transform
        transformed_data = vectorizer.transform([input_text])

        # Get all the feature name(words instead of numbers) corresponding to the array
        feature_names = vectorizer.get_feature_names_out()

        # calculate initial scores and store them in a dictionary
        scores_dict = {name: score for name, score in zip(feature_names, transformed_data.toarray()[0])}

        # adjust scores based on term length and update the dictionary
        for name in feature_names:
            scores_dict[name] *= (1 + 0.01 * len(name))

        # create a new array of adjusted scores in the same order as the feature names
        adjusted_scores = np.zeros(len(feature_names))
        for i, term in enumerate(feature_names):
            adjusted_scores[i] = scores_dict[term]

        # return a tuple of the feature names and adjusted scores
        return feature_names, adjusted_scores

    parse_dates = ['Product', 'Date received', 'Date sent to company']
    dtypes = {'Date received': str,
              'Product': "category",
              'Sub-product': "category",
              'Issue': "category",
              'Sub-issue': "category",
              'Consumer complaint narrative': str,
              'Company public response': str,
              'Company': "category",
              'State': "category",
              'ZIP code': str,
              'Tags': "category",
              'Consumer consent provided?': str,
              'Submitted via': "category",
              'Date sent to company': str,
              'Company response to consumer': str,
              'Timely response?': str,
              'Consumer disputed?': str,
              'Complaint ID': int}

    # Read the csv file
    DS1_data = pd.read_csv("TrainData.csv", low_memory=False, dtype=dtypes,
                           parse_dates=parse_dates, nrows=10000)

    # Count the amount of issue category's
    IssueCountNormalized = DS1_data["Issue"].value_counts(normalize=True)

    # create tf-idf vectorizer and fit to input text
    vectorizer = TfidfVectorizer(stop_words='english',
                                 token_pattern=r'\b[a-zA-Z]+\b',
                                 analyzer='word',
                                 use_idf=True,
                                 smooth_idf=True,
                                 norm=None, tokenizer=None,
                                 preprocessor=None)

    vectorizer.fit_transform(DS1_data['Consumer complaint narrative'])

    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    # Maak een lijst van alle woorden in de corpus
    feature_names = vectorizer.get_feature_names_out()

    Mortgage_Terms = ["Trouble", "payment", "process",
                      "Struggling", "pay", "mortgage",
                      "Applying", "mortgage", "refinancing",
                      "Loan", "servicing", "payments", "escrow", "account",
                      "Closing", "modification", "collection", "foreclosure",
                      "Settlement", "costs", "Credit", "Underwriting",
                      ]

    # user_input = input("What question do you want to categorize? ")
    def Classifiy_string(user_input):
        # Gets return all the words with a score in form of array that were send
        data = tfidf_custom_scoring(user_input)

        scores_dict = {name: score for name, score in zip(data[0], data[1])}

        # adjust scores based on term length and update the dictionary
        for name in data[0]:
            if name in Mortgage_Terms:
                scores_dict[name] *= (1 + 0.01 * len(name) + 1.5)
            else:
                scores_dict[name] *= (1 + 0.01 * len(name))

        # create a new array of adjusted scores in the same order as the feature names
        adjusted_scores = np.zeros(len(feature_names))
        for i, term in enumerate(feature_names):
            adjusted_scores[i] = scores_dict[term]

        # Set the top 3 to a list.
        Top3Words = []

        # Get the index of the highest scoring word.
        index_max = adjusted_scores.argmax()

        def Get_Top3_Words():
            # Krijg de index van het hoogste cijfer in de array met scores
            index_max = adjusted_scores.argmax()

            while True:
                if index_max == 0 or len(Top3Words) >= 3:
                    break
                # Check if there are any vowels in the topword, if not select new one
                while True:
                    vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
                    if any(char in vowels for char in feature_names[index_max]):
                        break
                    else:
                        adjusted_scores[index_max] = 0
                        index_max = adjusted_scores.argmax()

                # If the word isn't found in the wordbank, select the next word
                while True:
                    if len(wordnet.synsets(feature_names[index_max])) != 0:
                        break
                    else:
                        adjusted_scores[index_max] = 0
                        index_max = adjusted_scores.argmax()

                # Krijg het bijbehorende woord
                top_word = str(data[0][index_max])

                # If the word is in the english vocabluary, add it to the top 3 list.
                if top_word in english_vocab:
                    Top3Words.append(top_word)

                # Sets the current score of this word to 0 to select the second most popular word
                adjusted_scores[index_max] = 0
                index_max = adjusted_scores.argmax()

        Get_Top3_Words()

        def Checkifcorrospondingword(Relevant_word):
            return DS1_data[DS1_data["top_word"].str[0].isin(Relevant_word)]

        filtered_df = Checkifcorrospondingword(Top3Words)

        # count the occurrences of each issue and get the most common one
        while True:
            # Checks if there are any top words defined left or all used.
            if len(Top3Words) == 0:
                return ""

            # Checks if tere are any history cases found like this one to compare, if not get 3 new top words.
            if filtered_df["Issue"].value_counts()[0] != 0:
                fitted_vectorizer = vectorizer.fit(DS1_data["Consumer complaint narrative"])
                tfidf_vectorizer_vectors = fitted_vectorizer.transform(DS1_data["Consumer complaint narrative"])

                model = LinearSVC().fit(tfidf_vectorizer_vectors, DS1_data["Issue"])

                prediction = model.predict(fitted_vectorizer.transform([user_input]))
                print(prediction)
                return prediction
            else:
                value_counts = filtered_df["Issue"].value_counts()
                NormalizedTable = pd.concat([IssueCountNormalized,
                                             value_counts],
                                            axis=1,
                                            keys=('perc', 'valuecount'))
                Endscores = NormalizedTable.valuecount / NormalizedTable.perc
                NormalizedTable["Endscores"] = Endscores
                NormalizedTable["IssueName"] = IssueCountNormalized.keys().tolist()

                Toprow = NormalizedTable.loc[NormalizedTable['Endscores'].idxmax()]
                print(Toprow)
                return Toprow.IssueName
    return Classifiy_string(user_input)

def generateMetrics():
    return "Category was: " + LoadData()

class Reply(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # initialize

        parser.add_argument('question', required=True)  # add arguments

        args = parser.parse_args()  # parse arguments to dictionary
        # create new dataframe containing new values
        return {'data': args['question'] + " read"}, 200  # return data with 200 OK
    def get(self):
        response = make_response(generateMetrics(), 200)
        response.mimetype = "text/plain"
        return response

api.add_resource(Reply, '/reply')  # '/users' is our entry point for Users

if __name__ == '__main__':
    app.run()  # run our Flask app