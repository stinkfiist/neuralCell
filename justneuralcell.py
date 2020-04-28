from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #program her çalıştığında aynı sayıları üretir.

        random.seed(1)

        #tek nöron modelliyoruz, 3 doğru girdi ve 1 çıktı bağlantısıyla
        #aralığı -1 den 1e ve ortalaması sıfır 3x1 rastgele matrix atıyoruz.
        self.synaptic_weights = 2 * random.random((3, 1)) -1

    #sigmoid fonksiyonu S şekilli eğri tanımlar;
    #Bu fonksiyon ile girdilerin ağırlıklı toplamını 0 ile 1 arasında
    #normalize etmek için geçiyoruz.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid fonksiyonunun türevi, sigmoid eğrisinin gradyanıdır.
    # Mevcut ağırlıkla ilgili ne kadar doğru olduğunu gösterir.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    ## Sinir ağını bir deneme yanılma süreci ile eğitiyoruz.
    # Sinaptik ağırlıkların her seferinde ayarlanması amacıyla
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output= self.think(training_set_inputs)

            #hatayı hesapla:
            error = training_set_outputs - output

            #girdiyle hatayı çarp sonra bir de Sigmoid eğrisinin gradienti ile çarp
            #daha az güvenli ağırlıkların daha fazla ayarlanması anlamına gelir.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #ağırlıkları uydur:
            self.synaptic_weights += adjustment

    # Nöral ağ düşünmesi:
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptik weights: ")
    print (neural_netwotk.synaptic_weights)

    #Eğitme seti; 4 örnek var birisi çıktı, üçü girdi:
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #training set kullanarak nöral networku eğit.
    #10.000 defa yap ve her seferinde küçük uydurmalar yaptır
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
