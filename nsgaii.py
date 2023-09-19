import math
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_VIENNET = 0
NUM_PADRES = 100
NUM_HIJOS = 100
NUM_GENERACIONES = 500
FACTOR_MUTACION = .5 #1/n

viennet = [
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: x**2 + (y -1)**2,
            lambda x,y: x**2 + (y+1)**2 + 1,
            lambda x,y: (x-1)**2+y*2+2
        ],
        "lim_inf": -2,
        "lim_sup": 2
    },
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: (((x-2)**2)/2) + (((y+1)**2)/13),
            lambda x,y: (((x+y-3)**2)/36) + (((-x+y+2)**2)/8) - 17,
            lambda x,y: (((x+(2*y)-1)**2)/175) + ((((2*y)-x)**2)/17) -13
        ],
        "lim_inf": -4,
        "lim_sup": 4
    },
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: (.5*(x**2) + (y**2)) + math.sin((x**2) + (y**2)),
            lambda x,y: (((3*x -2*y +4)**2)/8) + (((x-y+1)**2)/27) + 15,
            lambda x,y: (1/((x**2)+(y**2)+1)) + 1.1*math.exp((-x**2) - (y**2))
        ],
        "lim_inf": -3,
        "lim_sup": 3
    }
]

class Individuo:
    def __init__(self, x=None, y=None):
        lim_inf = viennet[NUM_VIENNET]['lim_inf']
        lim_sup = viennet[NUM_VIENNET]['lim_sup']

        if(x is None):
            self.x= self.escoger_numero_random()
        else:
            if(x > lim_sup):
                self.x = lim_sup
            elif (x < lim_inf):
                self.x = lim_inf
            else:
                self.x = x

        if(y is None):
            self.y= self.escoger_numero_random()
        else:
            if(y > lim_sup):
                self.y = lim_sup
            elif (y < lim_inf):
                self.y = lim_inf
            else:
                self.y = y

        self.calcular_valores_objetivo()
        
    def calcular_valores_objetivo(self):
        valores = []
        for funcion in viennet[NUM_VIENNET]['funciones']:
            valores.append(funcion(self.x, self.y))
        self.val_obj = valores


    def escoger_numero_random(self):
        funcion = viennet[NUM_VIENNET]
        return random.uniform(funcion['lim_inf'], funcion['lim_sup'])

    
def x_domina_y(ind_x, ind_y):
    for funcion in viennet[NUM_VIENNET]['funciones']:
        if (not funcion(ind_x.x, ind_x.y) <= funcion(ind_y.x, ind_y.y)):
            return False
        
    for funcion in viennet[NUM_VIENNET]['funciones']:
        if (funcion(ind_x.x, ind_x.y) < funcion(ind_y.x, ind_y.y)):
            return True
    
    return False


def generar_poblacion(longitud_poblacion):
    poblacion = []

    for _ in range(longitud_poblacion):
        poblacion.append(Individuo())

    return poblacion
 
def fast_non_dominated_sort(P):
    F = []
    frente_actual = []

    for p in P:
        p.contador_dominacion = 0
        p.set_dominados = set()
        for q in P:
            if x_domina_y(p,q):
                p.set_dominados.add(P.index(q))
            elif x_domina_y(q,p):
                p.contador_dominacion +=1
        if p.contador_dominacion == 0:
            p.rango = 1
            frente_actual.append(p)
    
    while(frente_actual):
        sig_frente = []
        for p in frente_actual:
            for q_index in p.set_dominados:
                P[q_index].contador_dominacion -= 1
                if(P[q_index].contador_dominacion == 0):
                    P[q_index].rango = p.rango + 1
                    sig_frente.append(P[q_index])
                
        F.append(frente_actual)
        frente_actual = sig_frente

    return F

def crowding_distance_assigment(frente):
    for i in frente:
        i.distancia = 0
    for i in range(1, len(viennet[NUM_VIENNET]["funciones"])):
        f_max=0
        f_min=0
        lista_valores = []
        for ind in frente:
            lista_valores.append(ind.val_obj[i])
        f_max = max(lista_valores)
        f_min = min(lista_valores)

        frente = sorted(frente, key=lambda ind: ind.val_obj[i])
        frente[0].distancia = float("inf")
        frente[-1].distancia = float("inf")
        for j in range(1, len(frente)-1):
            frente[j].distancia += (frente[j+1].val_obj[i] - frente[j-1].val_obj[i])/(f_max - f_min)

        return frente

def sbx_crossover(padre1, padre2):
    mu = random.uniform(0, 1)
    index_dist = 20

    if(mu <=.5):
        beta = (1/(2*(1-mu))) **(1/(index_dist+1))
    else:
        beta = (2*mu)**(1/(index_dist+1))

    hijo1x = .5*((1+beta)*padre1.x + (1-beta)*padre2.x)
    hijo1y = .5*((1+beta)*padre1.y + (1-beta)*padre2.y)

    hijo2x = .5*((1-beta)*padre2.x + (1+beta)*padre2.x)
    hijo2y = .5*((1-beta)*padre2.y + (1+beta)*padre2.y)

    return Individuo(hijo1x, hijo1y), Individuo(hijo2x, hijo2y)

def polynomial_mutation(ind):
    mu = random.uniform(0, 1)
    eta = 20

    if(mu >=.5):
        delta = (1 -(2*(1-mu)))**(1/(eta+1))
    else:
        delta = (2*mu)**(1/(eta+1)) - 1

    ind_x = ind.x + (viennet[NUM_VIENNET]['lim_sup'] - viennet[NUM_VIENNET]['lim_inf'])*delta
    ind_y = ind.y + (viennet[NUM_VIENNET]['lim_sup'] - viennet[NUM_VIENNET]['lim_inf'])*delta

    return Individuo(ind_x, ind_y)


def seleccion_torneo(poblacion):
    candidatos_torneo = random.sample(poblacion, 2)
    ganador = min(candidatos_torneo, key=lambda individuo: individuo.rango)

    return ganador

def aplicar_mutacion(individuo):
    if random.random() < FACTOR_MUTACION:
        return polynomial_mutation(individuo)
    else:
        return individuo
    

def generar_hijos(poblacion):
    hijos = []

    while(len(hijos) <= NUM_HIJOS):
        padre1 = seleccion_torneo(poblacion)
        padre2 = seleccion_torneo(poblacion)
        hijo1, hijo2 = sbx_crossover(padre1, padre2)

        hijos.append(aplicar_mutacion(hijo1))
        hijos.append(aplicar_mutacion(hijo2))

    return hijos


if __name__ == "__main__":
    contador_generacion = 1

    poblacion_padres = generar_poblacion(NUM_PADRES)
    poblacion_hijos = []

    vectores_objetivo = []

    while(contador_generacion < NUM_GENERACIONES):
        vectores_objetivo = []
        print("Generacion", contador_generacion)
        Rt = poblacion_padres + poblacion_hijos
        F = fast_non_dominated_sort(Rt)
        
        sig_padres = []
        for frente in F:
            frente = crowding_distance_assigment(frente)
            if(len(sig_padres) + len(frente) <= NUM_PADRES):
                sig_padres.extend(frente)
            else:
                frente = sorted(frente, key=lambda x:x.distancia, reverse=True)
                sig_padres.extend(frente[0:NUM_PADRES-len(sig_padres)])
                break
        
        for ind in Rt:
            vectores_objetivo.append(ind.val_obj)

        poblacion_padres = sig_padres
        poblacion_hijos =  generar_hijos(poblacion_padres)

        contador_generacion +=1
    
    x, y, z = zip(*vectores_objetivo)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, z, c='b', marker='o')    
    ax.set_xlabel('f1(x,y)')
    ax.set_ylabel('f2(x,y)')
    ax.set_zlabel('f3(x,y)')    
    plt.show()




    

    

    
            

