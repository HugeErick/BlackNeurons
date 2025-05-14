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

def is_palindrome(n):
    s = str(abs(n))
    return s == s[::-1]

def is_armstrong(n):
    s = str(abs(n))
    num_digits = len(s)
    return n == sum(int(d) ** num_digits for d in s)

def is_harshad(n):
    if n == 0:
        return False
    digit_sum = sum(int(d) for d in str(abs(n)))
    return n % digit_sum == 0

def is_square_free(n):
    if n == 0:
        return False
    for i in range(2, int(abs(n) ** 0.5) + 1):
        if n % (i * i) == 0:
            return False
    return True

def is_abundant(n):
    if n < 12:
        return False
    return sum(get_divisors(n)[:-1]) > n

def is_deficient(n):
    if n < 1:
        return False
    return sum(get_divisors(n)[:-1]) < n

def is_happy(n):
    seen = set()
    n = abs(n)
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1

def is_triangular(n):
    if n < 1:
        return False
    k = int((2 * n) ** 0.5)
    return k * (k + 1) // 2 == n

def is_catalan(n):
    # Catalan numbers: C_0 = 1, C_{k+1} = C_k * 2*(2k+1)/(k+2)
    if n < 1:
        return False
    c = 1
    k = 0
    while c < n:
        k += 1
        c = c * 2 * (2 * k - 1) // (k + 1)
    return c == n
