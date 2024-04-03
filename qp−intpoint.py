import numpy as np
# Calcula x, la solucion optima del problema.
    # Min (1/2) ∗ xT Qx + cT x
    # Sujeto a F x ≥ d
# La tolerancia para las condiciones necesarias de primer orden es
# tol = 1e − 05.
# Numero maximo de iteraciones, maxiter = 100.
# Devuelve: 
    # x vector soluci ́on del problema
    # μ multiplicador de Lagrange
    # iter n ́umero de iteraciones
def qp_intpoint(Q, c, F, d, iter = 100, tol = 1e-5, decreasing_factor = 0.5):
    m, n = F.shape
    # v represents vector of all variables [x, z, mu]
    # the initial guess must have z, mu > 0
    v = np.ones((n + 2*m, 1))

    def FNewton(v_k):
        # x is the first n entries
        # z is the next m entries
        # mu is the next m entries
        x, z, mu = np.split(v_k, [n, n+m, n+m+m])[:-1]
        complementary_measure = ((z.T@mu)/m)[0][0]
        disturbance = decreasing_factor*complementary_measure
        return np.block([[Q@x + c - F.T@mu], 
                         [np.multiply(z, mu) - disturbance],
                         [-F@x + z + d]])
    
    # The Jacobian of the matrix given by the FNewton() function
    def Jacobian_FNewton(v_k):
        Z = np.diag(v_k[n:n+m, :].flatten())
        MU = np.diag(v_k[n+m:n+m+m, :].flatten())
        return np.block(
                   [[Q               , np.zeros((n, m)) , -F.T],
                    [np.zeros((m, n)), MU               , Z],
                    [-F              , np.eye(m)        , np.zeros((m, m)) ]])
    
    def get_delta_v(v_k, F_vk):
        J = Jacobian_FNewton(v_k)
        adjustedF = F_vk
        adjustedJ = J
        z = v_k[n: n+m, :].flatten()
        ZInv = np.diag(1/z)
        adjustedF[n:n+m, :] = ZInv @ adjustedF[n:n+m, :]
        adjustedJ[n:n+m, :] = ZInv @ adjustedJ[n:n+m, :]
        return np.linalg.solve(adjustedJ, -F_vk)
    
    def get_alpha(v_k, delta_v_k):
        alpha = 1
        for i in range(n, n+m+m):
            if delta_v_k[i, 0] < 0:
                alpha = min(alpha, -v_k[i, 0]/delta_v_k[i, 0])
        return alpha*0.95

    it = 0
    current_F = FNewton(v)
    while (np.linalg.norm(current_F.flatten())**2 > tol) and it < iter:
        delta_v = get_delta_v(v, current_F) 
        alpha = get_alpha(v, delta_v)
        v += alpha*delta_v
        current_F = FNewton(v)
        it += 1
        decreasing_factor *= 1/2
    return v[0:n, :], v[n+m:n+2*m, :], it

n = 2
# We generate Q so that it is positive definite
Q = np.eye(n)
# Q = np.random.rand(n, n)
# V, R = np.linalg.qr(Q)
# Q = R.T @ R

#c = np.random.randn(n, 1)
c = np.zeros((n, 1))
#A = np.random.rand(m, n)
A = np.array([[-1,0], [0, -1], [1, 0], [0, 1]])
b = np.array([[-1], [-1], [1], [1]])
#print("F", F, "d", d)
#print(Q, c, F, d)
x = qp_intpoint(Q, c, A, b)
#x = quadratic_interior_method(Q, c, A, b, np.zeros((1, n)), np.zeros((1,1)))
print(x)