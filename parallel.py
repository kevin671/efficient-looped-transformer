"""
F: transformer
    F(x) = x + f(x)

F_loop: looped transformer
    for _ in num_loops:
        x = F(x)

Parallelization as Picard Iteration
x0, x1, .... xT (xk+1=F(xk)) (T=num_loops, such as 1,000) 

Initialize x0, x1, .... xT

While:
    For k=0 to T: (this can be parallelized)
        xk = x0 + 1/T(f(x0)+f(x1)+...+f(xk))

"""