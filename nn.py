import numpy as num
from matplotlib import pyplot as mpl, cm

def sigmoide(x):   
    return 1/(1+num.exp(-x))
    
def dsigmoide(x):
    y = sigmoide(x)
    return y*(1-y)

def tanh(x):   
    return num.tanh(x)
    
def dtanh(x):
    y = tanh(x)
    return 1-y*y

def relu(x):
	return num.maximum(0,x)
	
def drelu(x):
	return num.maximum(0,1)
    	
"""
def transformar_target(y):
    return num.array([[int(i==j) for j in num.unique(y)] for i in y])
"""

class Red_Neuronal():
    def __init__(self, S): 		
        num.random.seed(1000)				
        self.n = len(S) 	
        self.S = S
        self.w = [num.random.normal(0,m**(-0.5),(m+1, n)) for m, n in zip(S[:-1], S[1:])] 
        self.E = []
        self.P = []
#        mpl.matshow(self.w[0])
#        mpl.title(f"Peso w{0}")
#        mpl.show()	
    
    #Agregamos -1 a la última columna de la matriz
    def bias_add(self, V):    
        bias = -num.ones((len(V),1))
        return num.concatenate((V,bias),axis=1)
    
    #Quitamos los umbrales (última columna)
    def bias_sub(self, V):
        return V[:,:-1]

    def activacion(self,x):
        """Retorna la salida de la activación cada capa de la red neuronal"""
        #z = [num.zeros((1,self.S[k]+1)) for k in range(1,self.n)]
        y = [self.bias_add(x)]
        #y = [x]
        for k in range(self.n-2):
            output = self.bias_add(relu(num.dot(y[k], self.w[k])))
            y.append(output)
        y.append(tanh(num.dot(y[-1], self.w[-1])))
        return y

    def backpropagation(self, x, z, lr):
        y = self.activacion(x)
        dw = [num.zeros_like(wi) for wi in self.w]
        error = z - y[-1]

        if self.n==2:
            delta = error * dtanh(y[1])
            dw[0] = lr * num.dot(y[0].T, delta)

        elif self.n==3:
            delta = error * dtanh(y[2])
            dw[1] = dw[1] + lr * num.dot(y[1].T, delta)

            error1 = num.dot( delta, self.w[1].T)
            delta1 = self.bias_sub(error1 * drelu(y[1]))
            dw[0] += lr * num.dot(y[0].T, delta1)

        elif self.n==4:
            delta = error * dtanh(y[3])
            dw[2] = dw[2] + lr * num.dot(y[3].T, delta)
            
            error2 = num.dot( delta, self.w[2].T)
            delta2 = self.bias_sub(error2 * drelu(y[2]))
            dw[1] += lr * num.dot(y[1].T, delta2)
            
            error1 = num.dot( delta2, self.w[1].T)
            delta1 = self.bias_sub(error1 * drelu(y[1]))
            dw[0] += lr * num.dot(y[0].T, delta1)

        return dw
        

    @staticmethod
    def loss(Z, Y):
        return num.mean( num.sum(( Z-Y)**2, axis=1))


    def adaptacion(self,dW):
        for w, dw in zip(self.W,dW):
            w += dw
        return self.W    
    

    def predicion(self, x):
        y = self.activacion(x)[-1]
        probs = y / num.sum(y, axis=1, keepdims=True)
        return num.argmax(probs, axis=1)
    
    
    def train(self, x, z, learning_rate, batch_size, parametro, epocas):
        P = len(x[:,0])
        H = num.random.permutation(P-batch_size)
        error = 1
        t = 1

        while (error>parametro) and (t<epocas):
            for h in H:
                Xh = x[h:h+batch_size]
                Zh = z[h:h+batch_size]
                dw = self.backpropagation(Xh, Zh, learning_rate)
                for k in range(len(self.w)):
                    self.w[k] = self.w[k] + dw[k]
                y = self.activacion(Xh)[-1]
                error = self.loss(Zh,y)*(1/P)
                prediccion = self.predicion(Xh)
                self.E.append(error)
                self.P.append(prediccion)
        t += 1
        if t%(int(epocas/10))==0:
            print("Epoca:",t," Error:",error)
        
  
    def info(self,x,z):
        y = self.activacion(x)[-1]
        loss = num.mean( num.square( z-y), axis=1)[0]
        print("Loss:",loss)
        print("Nuevos pesos:",self.w)
        mpl.plot(self.E)
        mpl.savefig('plot2.png')
        mpl.title('Learning rate=0.01 - Epocas=1000 - Datos entranamiento=70% \n S=[10,3,1]')
        mpl.show()

    
    def test(self, z, x):
    	return abs(z - self.bias_sub(self.activacion(x)))
    


#-------------------------------------------------------------------------------------------------
# Cargamos los datos
data_cancer = num.array(num.loadtxt('tp1_ej1_training.csv',delimiter=" "))
print("Dimension de los datos: ",data_cancer.shape)
"""
mpl.scatter( data_cancer[:,1], data_cancer[:,0], c=data_cancer[:,0].flatten(), cmap=cm.rainbow) 
mpl.colorbar()
mpl.title("Distribución de los datos (con una variable)")
mpl.xlabel("Radio")
mpl.ylabel("Diagnóstico \n <----  Maligno   --   Benigno  ---->")
mpl.show()
"""
# data_cancer[:,1:] #Radio, textura, perimetro, area, suavidad, compacidad, concavidad, puntos concavos, simetria
# data_cancer[:,0] #diagnóstico: maligno (0) - benigno (1)
total = len(data_cancer[:,0]) #total de datos
training = int(total*0.7)
test = total - training

test_set_inputs = num.array(data_cancer[training:,1:])
test_set_outputs = num.array(2*data_cancer[training:,0]-1)

training_set_inputs = num.array(data_cancer[:training,1:])
training_set_outputs = num.array(2*data_cancer[:training,0]-1)
print("inp:",training_set_inputs.shape,"\n out:",training_set_outputs.shape)

if __name__ == "__main__":
#    num.random.seed(100)
    #Creamos la red neuronal
    S = [10,3,1] # <---- ¿Cómo elegir?
    
    model1 = Red_Neuronal(S)
    
    #print("x: ",training_set_inputs)
    #print("z: ",training_set_outputs)
    
    model1.train(training_set_inputs, training_set_outputs,  0.05, 1, 0.05, 1000)

    y = model1.activacion(training_set_inputs)
    print(y)
    model1.info(training_set_inputs,training_set_outputs)

 #   print("Paso 1) Restauramos los pesos: ")
"""

      
    nro_entrenamiento = round(len(data_cancer[:,0])*0.8)
    training_set_inputs = data_cancer[:nro_entrenamiento,1:]
    training_set_outputs = 2*data_cancer[:nro_entrenamiento,0]-1
    
    test_set_inputs = data_cancer[nro_entrenamiento:,1:]
    test_set_outputs = 2*data_cancer[nro_entrenamiento:,0]-1
    
    print(training_set_outputs[:10])
 
    #print(num.array(training_set_inputs[0,:]).shape)
        
    #print(modelo1.correccion(data_cancer[:1,0], data_cancer[:1,1:], 0.1))

    
    #print(modelo1.correccion_simple(training_set_inputs, estimacion(test_set_outputs , (activacion_simple(Xh))[-1])))
    #E = modelo1.entrenamiento(training_set_inputs, training_set_outputs, 1, "simple", 30, 00.1, 0.001)
    #print(modelo1.entrenamiento(training_set_inputs, training_set_outputs, 1, "simple", 30, 0.1, 0.005))
    #mpl.plot(E, label="Error de estimación")
    #mpl.plot(Precision,label="Precisión")
    #mpl.xlabel("Nro. de entrenamiento")
    #mpl.ylabel("Error")
    #mpl.legend()
    #mpl.show()
    

    S2 = [10,5,1] 
    modelo2 = Red_Neuronal(S2)
    
    E2 = modelo2.entrenamiento(training_set_inputs, training_set_outputs, 10, "SGD", 30, 0.1, 0.05)
    mpl.plot(E2, label="Error de estimación")
    #mpl.plot(Precision,label="Precisión")
    mpl.xlabel("Nro. de entrenamiento")
    mpl.ylabel("Error")
    mpl.legend()
    mpl.show()    



    
    #print(training_set_inputs.shape)

 
    print("Paso 2) Nuevos Pesos luego del entrenamiento: ")
 #   neural_network.nuevos_pesos()

    # Test con una nueva situación.
    print("Paso 3) Testeamos un ejemplo EJ -> ?: ")
 #   capa_oculta_1, capa_oculta_2, output = neural_network.think(EJ)
#  print(output)
"""
