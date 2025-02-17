import numpy as np

# Coefficient matrix (A) and constant matrix (b) for Cramer's Rule
A_cramer = np.array([
    [2, 3, -1],
    [1, -2, 2],
    [-1, 4, 1]
], dtype=float)

b_cramer = np.array([4, 6, 5], dtype=float)

print("\n--- Coefficient Matrix (A) ---")
for row in A_cramer:
    print("  [", "  ".join(f"{val:6.2f}" for val in row), "]")


print("\n--- Constant Matrix (b) ---")
print("  [", "  ".join(f"{val:6.2f}" for val in b_cramer), "]")


# Solve using Cramer's Rule
def cramer_rule(A, b):
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("Determinant is zero, the system has no unique solution.")
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A
    return x

x_cramer = cramer_rule(A_cramer, b_cramer)
print("\n--- Solution using Cramer's Rule ---")
for i, xi in enumerate(x_cramer):
    print(f"x{i+1} = {xi:.4f}")

