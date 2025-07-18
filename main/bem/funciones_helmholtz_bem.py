import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.special import hankel1,jv
import matplotlib as mpl
from scipy.interpolate import griddata

# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 0,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    }

mpl.rcParams.update(pgf_with_latex)

def wavenumberToFrequency(k, c = 344.0):
    return 0.5 * k * c / np.pi

def frequencyToWavenumber(f, c = 344.0):
    return 2.0 * np.pi * f / c

def soundPressure(k, phi, t = 0.0, c = 344.0, density = 1.205):
    angularVelocity = k * c
    return (1j * density * angularVelocity  * np.exp(-1.0j*angularVelocity*t)
            * phi).astype(np.complex64)

def SoundMagnitude(pressure):
    return np.log10(np.abs(pressure / 2e-5)) * 20

def AcousticIntensity(pressure, velocity):
    return 0.5 * (np.conj(pressure) * velocity).real

def SignalPhase(pressure):
    return np.arctan2(pressure.imag, pressure.real)

def Normal2D(pointA, pointB):                  
    diff = pointA - pointB                          
    len = norm(diff)                                
    return np.array([diff[1]/len, -diff[0]/len]) 

def ComplexQuad(func, start, end):                                                 
    samples = np.array([[0.980144928249, 5.061426814519E-02],                           
                        [0.898333238707, 0.111190517227],                               
                        [0.762766204958, 0.156853322939],                               
                        [0.591717321248, 0.181341891689],                               
                        [0.408282678752, 0.181341891689],                               
                        [0.237233795042, 0.156853322939],                               
                        [0.101666761293, 0.111190517227],                               
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)    
    
    vec = end - start                                                                                                  
    sum = 0.0                                                                                                          
    for n in range(samples.shape[0]):                                                                                  
        x = start + samples[n, 0] * vec                                                                                
        sum += samples[n, 1] * func(x)                                                                                 
    return sum * norm(vec)                                                                                         
    
def ComputeL(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        if k == 0.0:                                                                                               
            ra = norm(p - qa)                                                                                      
            rb = norm(p - qb)                                                                                      
            re = norm(qab)                                                                                         
            return 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))                                        
        else:                                                                                                      
            def func(x):                                                                                           
                R = norm(p - x)                                                                                    
                return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)                                         
            return ComplexQuad(func, qa, p) + ComplexQuad(func, p, qa) \
                 + ComputeL(0.0, p, qa, qb, True)                                                              
    else:                                                                                                          
        if k == 0.0:                                                                                               
            return -0.5 / np.pi * ComplexQuad(lambda q: np.log(norm(p - q)), qa, qb)                           
        else:                                                                                                      
            return 0.25j * ComplexQuad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)                          
    return 0.0                                                                                                     
                                                                                                                   
def ComputeM(k, p, qa, qb, pOnElement):                                                                       
    qab = qb - qa                                                                                                  
    vecq = Normal2D(qa, qb)                                                                                    
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecq) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecq) / R                                                     
            return 0.25j * k * ComplexQuad(func, qa, qb)                                                       
    return 0.0                                                                                                 
                                                                                                                   
def ComputeMt(k, p, vecp, qa, qb, pOnElement):                                                                
    qab = qb - qa                                                                                                  
    if pOnElement:                                                                                                 
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecp) / np.dot(r, r)                                                              
            return -0.5 / np.pi * ComplexQuad(func, qa, qb)                                                    
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecp) / R                                                     
            return -0.25j * k * ComplexQuad(func, qa, qb)                                                      
                                                                                                                   
def ComputeN(k, p, vecp, qa, qb, pOnElement):                                                                 
    qab = qb- qa                                                                                                   
    if pOnElement:                                                                                                 
        ra = norm(p - qa)                                                                                          
        rb = norm(p - qb)                                                                                          
        re = norm(qab)                                                                                             
        if k == 0.0:                                                                                               
            return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re                                                
        else:                                                                                                      
            vecq = Normal2D(qa, qb)                                                                            
            k2 = k * k                                                                                             
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                R = np.sqrt(R2)                                                                                    
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                dpnu = np.dot(vecp, vecq)                                                                          
                c1 =  0.25j * k / R * hankel1(1, k * R)                                  - 0.5 / (np.pi * R2)      
                c2 =  0.50j * k / R * hankel1(1, k * R) - 0.25j * k2 * hankel1(0, k * R) - 1.0 / (np.pi * R2)      
                c3 = -0.25  * k2 * np.log(R) / np.pi                                                               
                return c1 * dpnu + c2 * drdudrdn + c3                                                              
            return ComputeN(0.0, p, vecp, qa, qb, True) - 0.5 * k2 * ComputeL(0.0, p, qa, qb, True) \
                 + ComplexQuad(func, qa, p) + ComplexQuad(func, p, qb)                                     
    else:                                                                                                          
        sum = 0.0j                                                                                                 
        vecq = Normal2D(qa, qb)                                                                                
        un = np.dot(vecp, vecq)                                                                                    
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                return (un + 2.0 * drdudrdn) / R2                                                                  
            return 0.5 / np.pi * ComplexQuad(func, qa, qb)                                                     
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)                                       
                R = norm(r)                                                                                        
                return hankel1(1, k * R) / R * (un + 2.0 * drdudrdn) - k * hankel1(0, k * R) * drdudrdn            
            return 0.25j * k * ComplexQuad(func, qa, qb)                                                       

def SolveLinearEquation(Ai, Bi, ci, alpha, beta, f):
    A = np.copy(Ai)
    B = np.copy(Bi)
    c = np.copy(ci)

    x = np.empty(c.size, dtype=complex)
    y = np.empty(c.size, dtype=complex)

    gamma = np.linalg.norm(B, np.inf) / np.linalg.norm(A, np.inf)
    swapXY = np.empty(c.size, dtype=bool)
    for i in range(c.size):
        if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
            swapXY[i] = False
        else:
            swapXY[i] = True

    for i in range(c.size):
        if swapXY[i]:
            for j in range(alpha.size):
                c[j] += f[i] * B[j,i] / beta[i]
                B[j, i] = -alpha[i] * B[j, i] / beta[i]
        else:
            for j in range(alpha.size):
                c[j] -= f[i] * A[j, i] / alpha[i]
                A[j, i] = -beta[i] * A[j, i] / alpha[i]

    A -= B
    y = np.linalg.solve(A, c)#scipy.sparse.linalg.lgmres(A, c)#np.linalg.solve(A, c)

    for i in range(c.size):
        if swapXY[i]:
            x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
        else:
            x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

    for i in range(c.size):
        if swapXY[i]:
            temp = x[i]
            x[i] = y[i]
            y[i] = temp

    return x, y

def computeBoundaryMatrices(k, mu, aVertex, aElement, orientation):
    A = np.empty((aElement.shape[0], aElement.shape[0]), dtype=complex)
    B = np.empty(A.shape, dtype=complex)

    for i in range(aElement.shape[0]):
        pa = aVertex[aElement[i, 0]]
        pb = aVertex[aElement[i, 1]]
        pab = pb - pa
        center = 0.5 * (pa + pb)
        centerNormal = Normal2D(pa, pb)
        for j in range(aElement.shape[0]):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, center, qa, qb, i==j)
            elementM  = ComputeM(k, center, qa, qb, i==j)
            elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
            elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
            
            A[i, j] = elementL + mu * elementMt
            B[i, j] = elementM + mu * elementN

        if orientation == 'interior':
            # interior variant, signs are reversed for exterior
            A[i,i] -= 0.5 * mu
            B[i,i] += 0.5
        elif orientation == 'exterior':
            A[i,i] += 0.5 * mu
            B[i,i] -= 0.5
        else:
            assert False, 'Invalid orientation: {}'.format(orientation)
            
    return A, B


def computeBoundaryMatricesExterior(k, mu, aVertex, aElement, orientation):
    orientation == 'exterior'
    A = np.empty((aElement.shape[0], aElement.shape[0]), dtype=complex)
    B = np.empty(A.shape, dtype=complex)

    for i in range(aElement.shape[0]):
        pa = aVertex[aElement[i, 0]]
        pb = aVertex[aElement[i, 1]]
        pab = pb - pa
        center = 0.5 * (pa + pb)
        centerNormal = Normal2D(pa, pb)
        for j in range(aElement.shape[0]):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, center, qa, qb, i==j)
            elementM  = ComputeM(k, center, qa, qb, i==j)
            elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
            elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
            
            A[i, j] = elementL + mu * elementMt
            B[i, j] = elementM + mu * elementN

        if orientation == 'interior':
            # interior variant, signs are reversed for exterior
            A[i,i] -= 0.5 * mu
            B[i,i] += 0.5
        elif orientation == 'exterior':
            A[i,i] += 0.5 * mu
            B[i,i] -= 0.5
        else:
            assert False, 'Invalid orientation: {}'.format(orientation)
            
    return A, B

def BoundarySolution(c, density, k, aPhi, aV):
    res = f"Density of medium:      {density} kg/m^3\n"
    res += f"Speed of sound:         {c} m/s\n"
    res += f"Wavenumber (Frequency): {k} ({wavenumberToFrequency(k)} Hz)\n\n"
    res += "index          Potential                   Pressure                    Velocity              Intensity\n"

    for i in range(aPhi.size):
        pressure = soundPressure(k, aPhi[i], t=0.0, c=344.0, density=1.205)
        intensity = AcousticIntensity(pressure, aV[i])
        res += f"{i+1:5d}  {aPhi[i].real: 1.4e}+ {aPhi[i].imag: 1.4e}i   {pressure.real: 1.4e}+ {pressure.imag: 1.4e}i   {aV[i].real: 1.4e}+ {aV[i].imag: 1.4e}i    {intensity: 1.4e}\n"
    
    return res

def solveInteriorBoundary(k, alpha, beta, f, phi, v, aVertex, aElement, c_=0, density=0, mu = None, orientation = 'interior'):
    mu = (1j / (k + 1))
    assert f.size == aElement.shape[0]
    A, B = computeBoundaryMatrices(k, mu, aVertex, aElement, orientation)
    c = np.empty(aElement.shape[0], dtype=complex)
    for i in range(aElement.shape[0]):
        # Note, the only difference between the interior solver and this
        # one is the sign of the assignment below.
        c[i] = phi[i] + mu * v[i]

    phi, v = SolveLinearEquation(B, A, c, alpha, beta, f)
    res = BoundarySolution(c_, density, k, phi, v)
    #print(res)
    return  v, phi

 
def solveExteriorBoundary(k, alpha, beta, f, phi, v, aVertex, aElement, c_=0, density=0, mu = None, orientation = 'exterior'):
    mu = (1j / (k + 1))
    assert f.size == aElement.shape[0]
    A, B = computeBoundaryMatrices(k, mu, aVertex, aElement, orientation)
    c = np.empty(aElement.shape[0], dtype=complex)
    for i in range(aElement.shape[0]):
        # Note, the only difference between the interior solver and this
        # one is the sign of the assignment below.
        c[i] = -(phi[i] + mu * v[i])

    phi, v = SolveLinearEquation(B, A, c, alpha, beta, f)
    #res = BoundarySolution(c_, density, k, phi, v)
    return v, phi


def solveSamples(k, aV, aPhi, aIncidentPhi, aSamples, aVertex, aElement, orientation):
    assert aIncidentPhi.shape == aSamples.shape[:-1], \
        "Incident phi vector and sample points vector must match"

    aResult = np.empty(aSamples.shape[0], dtype=complex)

    for i in range(aIncidentPhi.size):
        p  = aSamples[i]
        sum = aIncidentPhi[i]
        for j in range(aPhi.size):
            qa = aVertex[aElement[j, 0]]
            qb = aVertex[aElement[j, 1]]

            elementL  = ComputeL(k, p, qa, qb, False)
            elementM  = ComputeM(k, p, qa, qb, False)
            if orientation == 'interior':
                sum += elementL * aV[j] - elementM * aPhi[j]
            elif orientation == 'exterior':
                sum -= elementL * aV[j] - elementM * aPhi[j]
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
        aResult[i] = sum
    return aResult

def solveInterior(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, orientation = 'interior'):
    return solveSamples(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, orientation)

def solveExterior(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, orientation = 'exterior'):
    return solveSamples(k, aV, aPhi, aIncidentInteriorPhi, aInteriorPoints, aVertex, aElement, orientation)

def printInteriorSolution(k, c, density, aPhiInterior):
    print("\nSound pressure at the sample points\n")
    print("index          Potential                    Pressure               Magnitude         Phase\n")
    for i in range(aPhiInterior.size):
        pressure = soundPressure(k, aPhiInterior[i], c=c, density=density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print("{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase))
        
def Square():
    aVertex = np.array([[0.00, 0.0000], [0.00, 0.0125], [0.00, 0.0250], [0.00, 0.0375],
                         [0.00, 0.0500], [0.00, 0.0625], [0.00, 0.0750], [0.00, 0.0875],
                         
                         [0.0000, 0.10], [0.0125, 0.10], [0.0250, 0.10], [0.0375, 0.10],
                         [0.0500, 0.10], [0.0625, 0.10], [0.0750, 0.10], [0.0875, 0.10],
                         
                         [0.10, 0.1000], [0.10, 0.0875], [0.10, 0.0750], [0.10, 0.0625],
                         [0.10, 0.0500], [0.10, 0.0375], [0.10, 0.0250], [0.10, 0.0125],
                         
                         [0.1000, 0.00], [0.0875, 0.00], [0.0750, 0.00], [0.0625, 0.00],
                         [0.0500, 0.00], [0.0375, 0.00], [0.0250, 0.00], [0.0125, 0.00]], dtype=np.float32)

    aEdge = np.array([[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
                      [ 4,  5], [ 5,  6], [ 6,  7], [ 7,  8],
                      
                      [ 8,  9], [ 9, 10], [10, 11], [11, 12],
                      [12, 13], [13, 14], [14, 15], [15, 16],
                      
                      [16, 17], [17, 18], [18, 19], [19, 20],
                      [20, 21], [21, 22], [22, 23], [23, 24],
                      
                      [24, 25], [25, 26], [26, 27], [27, 28],
                      [28, 29], [29, 30], [30, 31], [31,  0]], dtype=np.int32)

    return aVertex, aEdge
def Square_n(n=10, length=0.1):

    h = length / n

    # Generar puntos por lado (sin repetir esquinas)
    left   = [(0.0, i * h) for i in range(n)]                      # 0 → n-1
    top    = [(i * h, length) for i in range(n)]                   # n → 2n-1
    right  = [(length, length - i * h) for i in range(n)]          # 2n → 3n-1
    bottom = [(length - i * h, 0.0) for i in range(n)]             # 3n → 4n-1

    # Concatenar en orden deseado
    aVertex = np.array(left + top + right + bottom, dtype=np.float32)

    # Crear aristas conectando consecutivamente + cierre del contorno
    num_vertices = 4 * n
    aEdge = np.array([[i, (i + 1) % num_vertices] for i in range(num_vertices)], dtype=np.int32)

    return aVertex, aEdge

def Circle_n(n=40, radius=1.0):

    # Ángulos en sentido horario
    theta = np.linspace(0, -2 * np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    aVertex = np.vstack((x, y)).T.astype(np.float32)

    # Crear aristas conectando puntos consecutivos + cierre del contorno
    aEdge = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)

    return aVertex, aEdge


def generateInteriorPoints_test_problem_2(Nx=10, Ny=10, length=0.1):

    # Evitar incluir los bordes: desplazamos un poco desde 0 hasta length
    x = np.linspace(length / (Nx + 1), length * Nx / (Nx + 1), Nx)
    y = np.linspace(length / (Ny + 1), length * Ny / (Ny + 1), Ny)

    X, Y = np.meshgrid(x, y)
    interiorPoints = np.column_stack([X.ravel(), Y.ravel()])
    return interiorPoints.astype(np.float32)

def generateInteriorPoints_excluding_circle(Nx=5, Ny=5, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, r_exclude=1.0):

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Calcular distancia al origen
    distance_squared = points[:, 0]**2 + points[:, 1]**2

    # Máscara de puntos fuera y dentro del círculo
    mask_outside = distance_squared >= (r_exclude)**2
    mask_inside  = ~mask_outside

    points_outside = points[mask_outside].astype(np.float32)
    points_inside  = points[mask_inside].astype(np.float32)

    return points_outside, points_inside

def generateRectangleBoundaryPoints_excluding_circle(Nx=20, Ny=20,
                                                     xmin=-2.0, xmax=2.0,
                                                     ymin=-2.0, ymax=2.0,
                                                     r_exclude=1.0):
    # Bordes: izquierdo, derecho, inferior, superior
    x_left   = np.full(Ny, xmin)
    x_right  = np.full(Ny, xmax)
    y_bottom = np.full(Nx, ymin)
    y_top    = np.full(Nx, ymax)

    y_vals = np.linspace(ymin, ymax, Ny)
    x_vals = np.linspace(xmin, xmax, Nx)

    # Crear puntos sobre los 4 lados
    left   = np.column_stack((x_left, y_vals))
    right  = np.column_stack((x_right, y_vals))
    bottom = np.column_stack((x_vals, y_bottom))
    top    = np.column_stack((x_vals, y_top))

    # Unir todos los bordes
    all_edges = np.vstack((left, right, bottom, top))

    # Quitar duplicados en esquinas (opcional)
    all_edges = np.unique(all_edges, axis=0)

    # Filtrar puntos fuera del círculo
    distance_squared = all_edges[:, 0]**2 + all_edges[:, 1]**2
    mask_outside = distance_squared >= r_exclude**2

    boundary_points = all_edges[mask_outside].astype(np.float32)

    return boundary_points

def phi_test_problem_1_2(p1, p2, k):
    factor = k / np.sqrt(2)
    return np.sin(factor * p1) * np.sin(factor * p2)

def plot_edges_and_field(vertices, elementos, centros, f, cmap="magma"):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [1.4, 2]})

    # Subplot 1: Aristas orientadas con flechas
    ax = axs[0]
    for i, (start, end) in enumerate(elementos):
        p1, p2 = vertices[start], vertices[end]
        dx, dy = p2 - p1
        ax.arrow(p1[0], p1[1], dx, dy, head_width=0.06, length_includes_head=True, 
                 color='blue', alpha=0.7)

    # Calcular centroide para desplazar etiquetas hacia afuera
    centroide = np.mean(vertices, axis=0)

    for i, (x, y) in enumerate(vertices):
        ax.plot(x, y, 'ko', markersize=3)
        # Vector desde centroide hacia el nodo
        dx, dy = x - centroide[0], y - centroide[1]
        norma = np.sqrt(dx**2 + dy**2) + 1e-12
        offset_x, offset_y = 0.12 * dx / norma, 0.12 * dy / norma
        ax.text(x + offset_x, y + offset_y, f'{i}', color='red', fontsize=8, ha='center', va='center')
 
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(- 1.0, 1.0)
    #ax.grid(True) 
    #ax.grid(True)

    # Subplot 2: Campo escalar en centros
    ax = axs[1]
    colormap = plt.get_cmap(cmap)

    for i, (start, end) in enumerate(elementos):
        p1, p2 = vertices[start], vertices[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=2)

    sc = ax.scatter(centros[:, 0], centros[:, 1], c=f.real, cmap=colormap, edgecolors='k', zorder=3)

    cbar = fig.colorbar(sc, ax=ax, label="Condición de contorno", orientation="vertical", pad=0.02, aspect=20, shrink=1.0)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(- 1.0, 1.0)
    #ax.grid(True)
 
    plt.savefig("figs/edges_and_field.svg", dpi=150, bbox_inches='tight')
    plt.show()
    

def plot_solutions(exact_sol, num_sol, interiorPoints):
 
    # Crear figura con tres subgráficos lado a lado
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

    # Solución numérica
    tcf1 = axs[0].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], num_sol, levels=50, cmap='magma')
    fig.colorbar(tcf1, ax=axs[0], label=r'$u(p)$')
    axs[0].set_title("Solución Numérica")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Solución exacta
    tcf2 = axs[1].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], exact_sol, levels=50, cmap='magma')
    fig.colorbar(tcf2, ax=axs[1], label=r'$u(p)$')
    axs[1].set_title("Solución Exacta")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Error absoluto
    error = np.abs(num_sol - exact_sol)
    tcf3 = axs[2].tricontourf(interiorPoints[:, 0], interiorPoints[:, 1], error, levels=50, cmap='magma')
    fig.colorbar(tcf3, ax=axs[2], label=r'Error $|u_{num} - u_{exact}|$')
    axs[2].set_title("Error")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    plt.show()

# Function to compute the exact solution
def sound_hard_circle_calc(k0, a, X, Y, n_terms=None):
 
    points = np.column_stack((X.ravel(), Y.ravel()))
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    if n_terms is None:
        n_terms = int(30 + (k0 * a)**1.01)
    u_scn = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
        hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
        u_scn += (-(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
            np.exp(1j*n*theta)).ravel()
    u_scn = np.reshape(u_scn, X.shape)
    u_inc = np.exp(1j*k0*X)
    u = u_inc + u_scn
    return u_inc, u_scn, u


def mask_displacement(R_exact, r_i, r_e, u):
 
    u = np.ma.masked_where(R_exact < r_i, u)
    return u

def plot_exact_displacement(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
 

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.5  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{inc}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u_{\rm{sct}}$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, u_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u$")
    cb3.set_ticks([-1.5, 1.5])
    cb3.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{inc}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u_{\rm{sct}}$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, u_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"$u$")
    cb6.set_ticks([-(np.pi),(np.pi)])
    cb6.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'Exact - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'Exact - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figs/displacement_exact.svg", dpi=300, bbox_inches='tight')


def plot_bem_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.5  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, u_phase/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(u_phase)/np.abs(u_scn_phase).max()])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'BEM - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'BEM - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figs/displacement_bem.svg", dpi=150, bbox_inches='tight')


def plot_bem_displacement_error(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the normalized error of the amplitude and phase (last column only).

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_amp (numpy.ndarray): Total displacement field (complex).
    u_scn_amp (numpy.ndarray): Scattered displacement field (complex).
    u_phase (numpy.ndarray): Phase of the total displacement.
    u_scn_phase (numpy.ndarray): Phase of the scattered displacement.
    """

    fig, axs = plt.subplots(2, 1, figsize=(2.5, 3.5))
    shrink = 0.5

    # Normalized error in amplitude
    c1 = axs[0].pcolormesh(X, Y, np.abs(u_amp) / np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"|Error| / max($u$)")
    max_amp = np.max(np.abs(u_amp) / np.abs(u_scn_amp).max())
    cb1.set_ticks([0, max_amp])
    cb1.set_ticklabels([f'{0:.1f}', f'{max_amp:.4f}'], fontsize=7)
    axs[0].axis("off")
    axs[0].set_aspect("equal")

    # Normalized error in phase
    c2 = axs[1].pcolormesh(X, Y, u_phase / np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$)")
    max_phase = np.max(u_phase / np.abs(u_scn_phase).max())
    cb2.set_ticks([0, max_phase])
    cb2.set_ticklabels([f'{0:.1f}', f'{max_phase:.4f}'], fontsize=7)
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    # Add rotated labels
    fig.text(0.05, 0.80, r'BEM - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'BEM - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("figs/displacement_bem_error.svg", dpi=150, bbox_inches='tight')    