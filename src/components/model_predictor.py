import pickle

class ModelPredictor:
    def __init__(self):
        self.vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
        self.model = pickle.load(open("artifacts/model.pkl", "rb"))
    
    def predict(self, text):
        text = self.vectorizer.transform([text])
        return self.model.predict(text)[0]

if __name__ == "__main__":
    predictor = ModelPredictor()
    print(predictor.predict("what is his earphone brand the mic is sick shit"))