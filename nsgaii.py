import math
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_VIENNET = 1
NUM_PADRES = 100
NUM_HIJOS = 100
NUM_GENERACIONES = 500
FACTOR_MUTACION = .5 #1/n
FACTOR_CROSSOVER = .5

viennet = [
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: x**2 + (y -1)**2,
            lambda x,y: x**2 + (y+1)**2 + 1,
            lambda x,y: ((x-1)**2)+(y**2)+2
        ],
        "lim_inf": -2,
        "lim_sup": 2
    },
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: (((x-2)**2)/2) + (((y+1)**2)/13) + 3,
            lambda x,y: (((x+y-3)**2)/36) + (((-x+y+2)**2)/8) - 17,
            lambda x,y: (((x+(2*y)-1)**2)/175) + ((((2*y)-x)**2)/17) -13
        ],
        "lim_inf": -4,
        "lim_sup": 4
    },
    {
        "tipo": "min",
        "funciones": [
            lambda x,y: (.5*((x**2) + (y**2))) + math.sin((x**2) + (y**2)),
            lambda x,y: (((3*x -2*y +4)**2)/8) + (((x-y+1)**2)/27) + 15,
            lambda x,y: (1/((x**2)+(y**2)+1)) + 1.1*math.exp((-x**2)-(y**2))
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
    nvar_hijos = []
    nvar = [(padre1.x, padre2.x), (padre1.y, padre2.y)]

    if np.random.uniform() <= FACTOR_CROSSOVER:
        index_dist = 20
        for i in range(0, len(nvar)):
            if np.fabs(nvar[i][0] - nvar[i][1]) > 1.2e-7:
                if nvar[i][0] < nvar[i][1]:
                    y1 = nvar[i][0]
                    y2 = nvar[i][1]
                else:
                    y1 = nvar[i][1]
                    y2 = nvar[i][0]
                yl = viennet[NUM_VIENNET]["lim_inf"]
                yu = viennet[NUM_VIENNET]["lim_sup"]
                rnd = 0
                while rnd == 0:
                    rnd = np.random.uniform()

                betaq = np.power(2*rnd, 1.0/(index_dist+1)) if rnd <= .5 else np.power(2*rnd, 1.0/(index_dist+1))

                #que hace esto?
                rnd = 0
                while rnd == 0:
                    rnd = np.random.uniform()
                betaq = 1 if rnd <= 0.5 else -1*betaq

                rnd = 0
                while rnd == 0:
                    rnd = np.random.uniform()
                betaq = 1 if rnd <= 0.5 else betaq

                rnd = 0
                while rnd == 0:
                    rnd = np.random.uniform()
                    betaq = 1 if rnd > FACTOR_CROSSOVER else betaq

                c1 = 0.5*((y1 + y2) - betaq*(nvar[i][0] - nvar[i][1]))
                c2 = 0.5*((y1 + y2) + betaq*(nvar[i][0] - nvar[i][1]))

                c1 = yl if c1 < yl else c1
                c2 = yl if c2 < yl else c2
                c1 = yu if c1 > yu else c1
                c2 = yu if c2 > yu else c2

                if np.random.uniform() >= 0.5:
                    nvar_hijos.append((c2, c1))
                else:
                    nvar_hijos.append((c1, c2))
            else:
                nvar_hijos.append((nvar[i][0], nvar[i][1]))
    else:
        for i in range(0, len(nvar)):
            nvar_hijos.append((nvar[i][0], nvar[i][1]))

    return Individuo(nvar_hijos[0][0], nvar_hijos[0][1]), Individuo(nvar_hijos[1][0], nvar_hijos[1][1])

def polynomial_mutation(valor):
    mu = random.uniform(0, 1)
    eta = 20
    mut_pow = 1 / (eta + 1)

    lim_sup = viennet[NUM_VIENNET]["lim_sup"]
    lim_inf = viennet[NUM_VIENNET]["lim_inf"]

    delta1 = (valor - lim_inf)/(lim_sup - lim_inf)
    delta2 = (lim_sup - valor)/(lim_sup - lim_inf)

    if mu <= .5:
        xy = 1 - delta1
        val = 2*mu + (1 - 2*mu)*np.power(xy, eta + 1)
        deltaq = np.power(val, mut_pow) - 1
    else:
        xy = 1 - delta2
        val = 2*(1 -mu) + 2*(mu - .5)*np.power(xy, eta + 1)
        deltaq = 1 - np.power(val, mut_pow)

    """if(mu >=.5):
        delta = (1 -(2*(1-mu)))**mut_power
    else:
        #deltaq = np.power((2*mu)**mut_power - 1
        val = 2*mu + (1 - 2*mu) * np.power(1 - )
        deltaq = np.power(val, mut_pow) - 1"""

    #ind_x = ind.x + (viennet[NUM_VIENNET]['lim_sup'] - viennet[NUM_VIENNET]['lim_inf'])*delta
    #ind_y = ind.y + (viennet[NUM_VIENNET]['lim_sup'] - viennet[NUM_VIENNET]['lim_inf'])*delta

    #ind_x = max(min(ind_x + deltaq*(lim_sup - lim_inf), lim_sup), lim_inf)
    #ind_y = max(min(ind_y + deltaq*(lim_sup - lim_inf), lim_sup), lim_inf)

    return max(min(valor + deltaq*(lim_sup - lim_inf), lim_sup), lim_inf)


def seleccion_torneo(poblacion):
    candidatos_torneo = random.sample(poblacion, 2)
    ganador = min(candidatos_torneo, key=lambda individuo: individuo.rango)

    return ganador

def aplicar_mutacion(individuo):
    if random.random() < FACTOR_MUTACION:
        return Individuo(polynomial_mutation(individuo.x), polynomial_mutation(individuo.y))
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




    

    

    
            


