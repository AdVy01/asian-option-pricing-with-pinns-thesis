import numpy as np
from scipy.linalg import solve_banded


def vecer_fixed_asian_call(S0, K, T, r, sigma, M, N, theta=0.5):
    """
    Prices a Fixed-Strike Asian Call using the Vecer dimension reduction.
    q(t) = 1 - t/T
    """
    # Setup Grid
    dt = T / N
    z_min, z_max = -1.0, 1.0
    dz = (z_max - z_min) / M
    z = np.linspace(z_min, z_max, M + 1)
    nu = (dz**2) / dt
    
    # Initial Condition: u(T, z) = (z)^+
    u = np.maximum(z, 0)
    
    # Backward Induction
    for j in range(N - 1, -1, -1):
        t_curr = (j + 0.5) * dt
        q_j = 1 - (t_curr / T)
        
        # Grid indices for inner points i = 1 to M-1
        zi = z[1:M]
        
        # Let A = sigma^2 * (q_j - z_i)^2 and B = dz * r * (q_j - z_i)
        A = (sigma**2) * (q_j - zi)**2
        B = dz * r * (q_j - zi)
        
        # LHS (Time j) coefficients
        L_left  = theta * (A - B)
        L_mid   = -2 * (theta * A + nu)
        L_right = theta * (A + B)
        
        # RHS (Time j+1) coefficients
        R_left  = -(1 - theta) * (A - B)
        R_mid   = 2 * ((1 - theta) * A - nu)
        R_right = -(1 - theta) * (A + B)
        
        # Construct RHS vector b
        # rhs_i = R_left*u_{i-1} + R_mid*u_i + R_right*u_{i+1}
        rhs = R_left * u[0:M-1] + R_mid * u[1:M] + R_right * u[2:M+1]
        
        # Boundary Conditions
        # Dirichlet: u[0,j] = 0.
        
        # Linear Extrapolation: u[M,j] = 2*u[M-1,j] - u[M-2,j]
        L_mid[-1]  += 2 * L_right[-1]
        L_left[-1] -= L_right[-1]
        
        # Solve Tridiagonal System
        # Format for solve_banded: [upper, diag, lower]
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = L_right[:-1] # Upper diagonal
        ab[1, :]  = L_mid        # Main diagonal
        ab[2, :-1] = L_left[1:]  # Lower diagonal
        
        u_inner = solve_banded((1, 1), ab, rhs)
        
        # Update u-vector for the next time step (j-1)
        u[1:M] = u_inner
        u[0] = 0 
        u[M] = 2 * u[M-1] - u[M-2]
        
    # Vecer normalization: Price = S0 * u(0, z_0)
    # where z_0 = 1 - (K / S0)
    z_0 = 1 - (K / S0)
    price = S0 * np.interp(z_0, z, u)
    
    return max(price, 0)

def vecer_fixed_asian_call_adaptive(S0, K, T, r, sigma, M, N, theta=0.5, c=5.0):
    """
    Prices a Fixed-Strike Asian Call using the Vecer dimension reduction 
    with an adaptive (non-uniform) grid clustered around z = 0.
    
    Parameters:
    c (float): Grid stretching parameter. Higher 'c' means denser clustering 
               around the kink. (Typical values: 2.0 to 5.0)
    """
    dt = T / N
    z_min, z_max = -1.0, 1.0
    
    # 1. Adaptive Grid Construction (Sinh Transformation)
    # We want to cluster points around z_c = 0 (the kink in the terminal payoff)
    z_c = 0.0
    xi = np.linspace(0, 1, M + 1) # Uniform abstract grid
    
    asinh_max = np.arcsinh(c * (z_max - z_c))
    asinh_min = np.arcsinh(c * (z_min - z_c))
    
    # Transform to non-uniform physical grid
    z = z_c + (1.0 / c) * np.sinh(xi * asinh_max + (1 - xi) * asinh_min)
    
    # 2. Precompute non-uniform grid spacings
    h = np.diff(z)       # h[i] = z[i+1] - z[i]
    hL = h[:-1]          # left spacing for inner nodes
    hR = h[1:]           # right spacing for inner nodes
    
    # 3. Finite difference weights for 1st and 2nd derivatives on a non-uniform grid
    # u_z \approx W1_L * u_{i-1} + W1_M * u_i + W1_R * u_{i+1}
    W1_L = -hR / (hL * (hL + hR))
    W1_M = (hR - hL) / (hL * hR)
    W1_R = hL / (hR * (hL + hR))
    
    # u_{zz} \approx W2_L * u_{i-1} + W2_M * u_i + W2_R * u_{i+1}
    W2_L = 2.0 / (hL * (hL + hR))
    W2_M = -2.0 / (hL * hR)
    W2_R = 2.0 / (hR * (hL + hR))
    
    # Initial Condition: u(T, z) = (z)^+
    u = np.maximum(z, 0)
    
    # Backward Induction
    for j in range(N - 1, -1, -1):
        t_curr = (j + 0.5) * dt
        q_j = 1 - (t_curr / T)
        
        # Inner nodes
        zi = z[1:M]
        
        # PDE Coefficients: u_t + C2 * u_{zz} + C1 * u_z = 0
        C2 = 0.5 * (sigma**2) * (q_j - zi)**2
        C1 = r * (q_j - zi)
        
        # Spatial Operator L(u) = C2 * u_{zz} + C1 * u_z
        Op_L = C2 * W2_L + C1 * W1_L
        Op_M = C2 * W2_M + C1 * W1_M
        Op_R = C2 * W2_R + C1 * W1_R
        
        # LHS (Implicit part applied to u^{new}) -> (I - theta * dt * L)
        L_left  = -theta * dt * Op_L
        L_mid   = 1.0 - theta * dt * Op_M
        L_right = -theta * dt * Op_R
        
        # RHS (Explicit part applied to u^{old}) -> (I + (1 - theta) * dt * L)
        R_left  = (1.0 - theta) * dt * Op_L
        R_mid   = 1.0 + (1.0 - theta) * dt * Op_M
        R_right = (1.0 - theta) * dt * Op_R
        
        # Construct RHS vector
        rhs = R_left * u[0:M-1] + R_mid * u[1:M] + R_right * u[2:M+1]
        
        # Boundary Conditions
        # Dirichlet: u[0, j] = 0 (Left boundary is implicitly 0, nothing to add to RHS)
        
        # Linear Extrapolation for right boundary u[M, j]
        # In a non-uniform grid: u_M = (1 + alpha)*u_{M-1} - alpha*u_{M-2}
        alpha = hR[-1] / hL[-1]
        
        # Substitute the extrapolation directly into the last tridiagonal row
        L_mid[-1]  += (1 + alpha) * L_right[-1]
        L_left[-1] -= alpha * L_right[-1]
        
        # Solve Tridiagonal System
        ab = np.zeros((3, M - 1))
        ab[0, 1:]  = L_right[:-1] # Upper diagonal
        ab[1, :]   = L_mid        # Main diagonal
        ab[2, :-1] = L_left[1:]   # Lower diagonal
        
        u_inner = solve_banded((1, 1), ab, rhs)
        
        # Update u-vector for the next time step
        u[1:M] = u_inner
        u[0] = 0 
        u[M] = (1 + alpha) * u[M-1] - alpha * u[M-2]
        
    z_0 = 1 - (K / S0)
    price = S0 * np.interp(z_0, z, u)
    
    return max(price, 0)