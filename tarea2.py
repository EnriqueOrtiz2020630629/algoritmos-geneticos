import random

POBLACION_INICIAL=20
NUM_PADRES = 10
TAMANO_TORNEO = 2
NUM_GENERACIONES = 50
LIMITE_INFERIOR = 0
LIMITE_SUPERIOR = 1023
VALOR_OBJETIVO = 0
PORCENTAJE_MUTACION = .10

class Individuo:
    def __init__(self, valor=None):
        if(valor is None):
            self.valor= self.escoger_numero_random()
        else:
            self.valor = int(valor)
        
        self.calcular_aptitud()

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
        return random.randint(LIMITE_INFERIOR, LIMITE_SUPERIOR)
    
    def calcular_aptitud(self):
        self.aptitud = abs(funcion_objetivo(self.valor) - VALOR_OBJETIVO)

def generar_poblacion(longitud_poblacion):
    poblacion = set()

    for _ in range(longitud_poblacion):
        poblacion.add(Individuo())

    return poblacion

def seleccion_torneo(poblacion, tamano_torneo, numero_padres):
    seleccion = set()

    while len(seleccion) < numero_padres:
        candidatos_torneo = random.sample(poblacion, tamano_torneo)
        ganador = min(candidatos_torneo, key=lambda individuo: individuo.aptitud)
        seleccion.add(ganador)

    return seleccion

def cruza_dos_puntos(padre1, padre2):
    longitud_maxima_comun = min(len(padre1.get_repr_gray()), len(padre2.get_repr_gray()))

    punto1 = random.randint(0, longitud_maxima_comun-1)
    punto2 = random.randint(punto1 + 1, longitud_maxima_comun)

    hijo1 = Individuo(int(gray_a_binario(padre1.get_repr_gray()[:punto1] + padre2.get_repr_gray()[punto1:punto2] + padre1.get_repr_gray()[punto2:]), 2))
    hijo2 = Individuo(int(gray_a_binario(padre2.get_repr_gray()[:punto1] + padre1.get_repr_gray()[punto1:punto2] + padre2.get_repr_gray()[punto2:]), 2))

    return hijo1, hijo2

def cruza(poblacion):
    hijos = []
    lista_poblacion = list(poblacion)
    
    for i in range(0, len(lista_poblacion), 2):
        hijo1, hijo2 = cruza_dos_puntos(lista_poblacion[i], lista_poblacion[i+1])
        hijos.append(aplicar_mutacion(hijo1))
        hijos.append(aplicar_mutacion(hijo2))

    if len(lista_poblacion) % 2 == 1:
        hijos.append(lista_poblacion[-1])

    return set(hijos)

def mutacion_dos_puntos(individuo):
    repr_gray = list(individuo.get_repr_gray())
    len_gray = len(repr_gray)

    punto1 = random.randint(0, len_gray-1)
    punto2 = random.randint(0, len_gray-1)
    
    repr_gray[punto1] = '1' if repr_gray[punto1] == '0' else '0'
    repr_gray[punto2] = '1' if repr_gray[punto2] == '0' else '0'

    return Individuo(int(gray_a_binario(''.join(repr_gray)), 2))

def funcion_objetivo(x):
    return int(3*(x**2))

def gray_a_binario(gray):
    binary = [gray[0]]

    for i in range(1, len(gray)):
        bit = '1' if gray[i] == '1' else '0'
        binary_bit = '1' if bit != binary[i - 1] else '0'
        binary.append(binary_bit)

    return ''.join(binary)


def imprimir_poblacion(poblacion):
    num_individuo = 1
    for individuo in poblacion:
        print("Individuo: {} Valor:{} Binario: {} Gray: {}"
              .format(num_individuo,
                      individuo.valor,
                      individuo.get_repr_binaria(), 
                      individuo.get_repr_gray()))
        num_individuo += 1

def aplicar_mutacion(individuo):
    if random.random() < PORCENTAJE_MUTACION:
        return mutacion_dos_puntos(individuo)
    else:
        return individuo

def detectar_terminacion_temprana(poblacion):
    for individuo in poblacion:
        if individuo.aptitud == 0:
            return True
    
    return False

if __name__ == "__main__":
    generacion_actual = 0
    poblacion = generar_poblacion(POBLACION_INICIAL)

    while(generacion_actual < NUM_GENERACIONES):
        generacion_actual+=1
        pool_reproduccion = seleccion_torneo(poblacion, TAMANO_TORNEO, NUM_PADRES)
        hijos = cruza(pool_reproduccion)
        poblacion = set()
        poblacion.update(pool_reproduccion)
        poblacion.update(hijos)

        poblacion_ordenada = sorted(poblacion, key=lambda x:x.aptitud)
        mejor_individuo = poblacion_ordenada[0]
        print("Generacion ", generacion_actual, "Valor: ", mejor_individuo.valor)


        if(detectar_terminacion_temprana(poblacion)):
            break        

    print('Generacion: ', generacion_actual)
    imprimir_poblacion(poblacion)


    


#leer capitulo 2 tesis
#implementar nsga2
#sbx simulated binary crossover
#polynomial mutation

#fast non dominated sorting

#tesis
#evolutionary algorithm capitulo 1
#binnet
#https://www.youtube.com/watch?v=bYPDUKR6SV4&t=602s