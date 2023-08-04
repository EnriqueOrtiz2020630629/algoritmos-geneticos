import random
import sys

class Individuo:
    limite_inferior = 0
    limite_superior = 5000

    def __init__(self, valor=None):
        if(valor is None):
            self.valor= self.escoger_numero_random()
        else:
            self.valor =  (int(valor) 
                           if(self.limite_inferior <= valor <= self.limite_superior) 
                           else self.escoger_numero_random())

    def get_repr_binaria(self):
        return bin(self.valor)[2:]

    def get_repr_gray(self):
        codigo_gray = ''
        bit_previo = '0'

        for bit in self.get_repr_binaria():
            codigo_gray += str(int(bit) ^ int(bit_previo))
            bit_previo = bit

        return codigo_gray

    def escoger_numero_random(self):
        return random.randint(self.limite_inferior, self.limite_superior)

def generar_poblacion(longitud_poblacion):
    poblacion = set()

    for _ in range(longitud_poblacion):
        poblacion.add(Individuo())

    return poblacion

def imprimir_poblacion(poblacion):
    num_individuo = 1
    for individuo in poblacion:
        print("Individuo: {} Valor:{} Binario: {} Gray: {}"
              .format(num_individuo,
                      individuo.valor,
                      individuo.get_repr_binaria(), 
                      individuo.get_repr_gray()))
        num_individuo += 1


if __name__ == "__main__":
    if(len(sys.argv) >= 2):
        longitud_poblacion = int(sys.argv[1])
    else:
        longitud_poblacion = 10

    poblacion = generar_poblacion(longitud_poblacion)
    imprimir_poblacion(poblacion)



    

