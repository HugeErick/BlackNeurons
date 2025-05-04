"""
number_analysis.py
Provides various number analysis utilities for data analysts and AI neurons.
"""
import math

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_multiples(n, count=10):
    """Return the first 'count' multiples of n (excluding zero)."""
    return [n * i for i in range(1, count + 1)]

def get_divisors(n):
    """Return all positive divisors of n."""
    if n == 0:
        return []
    divisors = set()
    for i in range(1, int(abs(n) ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(abs(n) // i)
    return sorted(divisors)

def is_perfect_number(n):
    """Return True if n is a perfect number."""
    if n < 2:
        return False
    return sum(get_divisors(n)[:-1]) == n

def is_perfect_square(n):
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

def is_perfect_cube(n):
    if n < 0:
        root = int(round(abs(n) ** (1/3)))
        return -root * root * root == n
    root = int(round(n ** (1/3)))
    return root * root * root == n

def is_fibonacci(n):
    """A number is Fibonacci if and only if one or both of (5*n^2 + 4) or (5*n^2 - 4) is a perfect square."""
    if n < 0:
        return False
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)
