import numpy as np
from scipy.optimize import minimize

def ejercicio1():
    """
    Minimizar f(x1,x2) = x1^2 + 2x2^2
    sujeto a: x1 + 2x2 - 3 <= 0
    """
    def objective(x):
        return x[0]**2 + 2*x[1]**2
    
    def constraint1(x):
        return -(x[0] + 2*x[1] - 3)  # Convert to >= 0 form
  
    x0 = np.array([0.0, 0.0])
    cons = ({'type': 'ineq', 'fun': constraint1})
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    
    print("Ejercicio 1 Results:")
    print(f"x1 = {result.x[0]:.6f}")
    print(f"x2 = {result.x[1]:.6f}")
    print(f"Optimal value = {result.fun:.6f}")
    if hasattr(result, 'lagrangian_multipliers'):
        print(f"λ = {result.lagrangian_multipliers[0]:.6f}")
    else:
        print("Lagrange multipliers not available")
    return result

def ejercicio2():
    """
    Minimizar f(x) = x1^2 + x2^2
    sujeto a: x1 + x2 - 2 <= 0
              x1 >= 0
    """
    def objective(x):
        return x[0]**2 + x[1]**2
    
    def constraint1(x):
        return -(x[0] + x[1] - 2)  # Convert to >= 0 form
    
    def constraint2(x):
        return x[0]  # x1 >= 0
    
    x0 = np.array([0.0, 0.0])
    
    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2})
    
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    
    print("\nEjercicio 2 Results:")
    print(f"x1 = {result.x[0]:.6f}")
    print(f"x2 = {result.x[1]:.6f}")
    print(f"Optimal value = {result.fun:.6f}")
    if hasattr(result, 'lagrangian_multipliers'):
        print(f"λ1 = {result.lagrangian_multipliers[0]:.6f}")
        print(f"λ2 = {result.lagrangian_multipliers[1]:.6f}")
    else:
        print("Lagrange multipliers not available")
    return result

def ejercicio3():
    """
    Maximizar f(x1,x2) = 3x1 + 4x2
    sujeto a: x1^2 + x2^2 <= 9
              x1 >= 0
              x2 >= 0
    """
    def objective(x):
        return -(3*x[0] + 4*x[1])  # Negative for minimization
    
    def constraint1(x):
        return -(x[0]**2 + x[1]**2 - 9)  # Convert to >= 0 form
    
    def constraint2(x):
        return x[0]  # x1 >= 0
    
    def constraint3(x):
        return x[1]  # x2 >= 0
    
    # Initial guess
    x0 = np.array([1.0, 1.0])
    
    # Define constraints
    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3})
    
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    
    print("\nEjercicio 3 Results:")
    print(f"x1 = {result.x[0]:.6f}")
    print(f"x2 = {result.x[1]:.6f}")
    print(f"Optimal value = {-result.fun:.6f}")  # Negative because we minimized
    if hasattr(result, 'lagrangian_multipliers'):
        print(f"λ1 = {result.lagrangian_multipliers[0]:.6f}")
        print(f"λ2 = {result.lagrangian_multipliers[1]:.6f}")
        print(f"λ3 = {result.lagrangian_multipliers[2]:.6f}")
    else:
        print("Lagrange multipliers not available")
    return result

def ejercicio4():
    """
    Ejemplo de dualidad:
    Minimizar f(x) = x^2
    sujeto a: x - 1 <= 0
    """
    def objective(x):
        return x[0]**2
    
    def constraint1(x):
        return -(x[0] - 1)  # Convert to >= 0 form
    
    x0 = np.array([0.0])
    cons = ({'type': 'ineq', 'fun': constraint1})
    result = minimize(objective, x0, method='SLSQP', constraints=cons)
    
    print("\nEjercicio 4 Results:")
    print("Primal Problem:")
    print(f"x = {result.x[0]:.6f}")
    print(f"Optimal primal value = {result.fun:.6f}")
    if hasattr(result, 'lagrangian_multipliers'):
        lambda_value = result.lagrangian_multipliers[0]
        print(f"λ = {lambda_value:.6f}")
        # Calculate dual value
        dual_value = -lambda_value**2/4 - lambda_value
        print(f"Optimal dual value = {dual_value:.6f}")
    else:
        print("Lagrange multipliers not available")
    return result

if __name__ == "__main__":
    print("Resolviendo problemas de optimización con condiciones KKT...")
    ejercicio1()
    ejercicio2()
    ejercicio3()
    ejercicio4()
