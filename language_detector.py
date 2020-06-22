import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *


class LanguageDetector():

    def __init__(self,classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(1,2),
            max_features=1000,
            preprocessor=self._remove_noise
        )

    def _remove_noise(self,text):
        noise_pattern = re.compile("|".join(["http\S+","\@\w+","\#\w+","\d+"]))
        clean_text = re.sub(noise_pattern,"",text)
        return clean_text

    def features(self,X):
        return self.vectorizer.transform(X)

    def fit(self,X,y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X).toarray(),y)

    def predict(self,x):
        return self.classifier.predict(self.features([x]).toarray())

    def score(self,X,y):
        return self.classifier.score(self.features(X),y)

if __name__ == '__main__':
    # read data from data.csv
    data_file = open("data.csv",encoding="utf-8")
    lines = data_file.readlines()
    data_file.close()
    dataset = [(line.strip()[:-3],line.strip()[-2:]) for line in lines]
    # x:langeges_sentences
    # y:tagas
    x,y = zip(*dataset)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
    # print(x_train[:20])

    language_dectector = LanguageDetector(classifier=MultinomialNB())
    language_dectector.fit(x_train,y_train)
    print(language_dectector.score(x_test,y_test))

    # English
    print(language_dectector.predict("This is an English sentence"))
    # German
    print(language_dectector.predict("Das ist ein deutscher Satz."))
    # French
    print(language_dectector.predict("C 'est un français."))