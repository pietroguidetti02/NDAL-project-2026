import numpy as np
import time

#we use class approach to encapsulate the client and server logic for federated learning
class FLClient:
    def __init__(self, client_id, X_train, y_train):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def set_model(self, model):
        """Assisgns a compiled/initialized model to the client."""
        self.model = model

    def get_weights(self):
        """extracts the weights of the model in a format suitable for aggregation."""
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            return {'coefs_': [np.copy(c) for c in self.model.coefs_], 
                    'intercepts_': [np.copy(i) for i in self.model.intercepts_]}
        else:
            # Assume Keras Model (LSTM)
            return self.model.get_weights()

    def set_weights(self, weights):
        """Sets the weights of the model."""
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            self.model.coefs_ = [np.copy(c) for c in weights['coefs_']]
            self.model.intercepts_ = [np.copy(i) for i in weights['intercepts_']]
        else:
            self.model.set_weights(weights)

    def train(self, epochs=1, batch_size=256):
        """executes local training on the client's data for a specified number of epochs."""
        start_time = time.perf_counter()
        from sklearn.neural_network import MLPClassifier
        if isinstance(self.model, MLPClassifier):
            # partial_fit executes one epoch of training on the data, so we loop for the specified number of epochs
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
        """executes the Federated Averaging (FedAvg) mathematical operation on the tensors."""
        if model_type == 'mlp':
            # coefs are the weights of the model, so we need to average them across clients
            new_coefs = []
            for i in range(len(list_of_weights[0]['coefs_'])):
                layer_coef = np.mean([w['coefs_'][i] for w in list_of_weights], axis=0)
                new_coefs.append(layer_coef)
                
            #intercept are what the model uses to adjust the output of each neuron, so we need to average them as well
            new_intercepts = []
            for i in range(len(list_of_weights[0]['intercepts_'])):
                layer_intercept = np.mean([w['intercepts_'][i] for w in list_of_weights], axis=0)
                new_intercepts.append(layer_intercept)
                
            return {'coefs_': new_coefs, 'intercepts_': new_intercepts}
        else:
            # Keras (list of numpy arrays) more efficiently handled with numpy
            new_weights = []
            for i in range(len(list_of_weights[0])):
                layer_w = np.mean([w[i] for w in list_of_weights], axis=0)
                new_weights.append(layer_w)
            return new_weights
