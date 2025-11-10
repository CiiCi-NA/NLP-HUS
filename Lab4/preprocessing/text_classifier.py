from sklearn.naive_bayes import MultinomialNB

class TextClassifier:
    def __init__(self, vectorizer, model_type='logistic'):
        self.vectorizer = vectorizer
        self.model_type = model_type
        self._model = None

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        if self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(solver='liblinear')
        elif self.model_type == 'nb':
            self._model = MultinomialNB()
        self._model.fit(X, labels)
