import numpy as np
import time

class FLClient:
    def __init__(self, client_id, X_train, y_train):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def set_model(self, model):
        """Assegna un modello compilato/inizializzato al client."""
        self.model = model

    def get_weights(self):
        """Estrae i pesi dal modello."""
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            return {'coefs_': [np.copy(c) for c in self.model.coefs_], 
                    'intercepts_': [np.copy(i) for i in self.model.intercepts_]}
        else:
            # Assume Keras Model (LSTM)
            return self.model.get_weights()

    def set_weights(self, weights):
        """Sovrascrive i pesi del modello."""
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            self.model.coefs_ = [np.copy(c) for c in weights['coefs_']]
            self.model.intercepts_ = [np.copy(i) for i in weights['intercepts_']]
        else:
            self.model.set_weights(weights)

    def train(self, epochs=1, batch_size=256):
        """Esegue il training locale sui dati del client."""
        start_time = time.perf_counter()
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            # partial_fit esegue esattamente 1 epoca (1 iterazione) sui dati forniti
            for _ in range(epochs):
                self.model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))
        else:
            # Keras LSTM
            self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        end_time = time.perf_counter()
        return end_time - start_time


class FLServer:
    def __init__(self):
        pass

    def aggregate_weights(self, list_of_weights, model_type='mlp'):
        """Esegue il Federated Averaging (FedAvg) matematico sui tensori."""
        if model_type == 'mlp':
            new_coefs = []
            for i in range(len(list_of_weights[0]['coefs_'])):
                layer_coef = np.mean([w['coefs_'][i] for w in list_of_weights], axis=0)
                new_coefs.append(layer_coef)
                
            new_intercepts = []
            for i in range(len(list_of_weights[0]['intercepts_'])):
                layer_intercept = np.mean([w['intercepts_'][i] for w in list_of_weights], axis=0)
                new_intercepts.append(layer_intercept)
                
            return {'coefs_': new_coefs, 'intercepts_': new_intercepts}
        else:
            # Keras (lista di array numpy)
            new_weights = []
            for i in range(len(list_of_weights[0])):
                layer_w = np.mean([w[i] for w in list_of_weights], axis=0)
                new_weights.append(layer_w)
            return new_weights
