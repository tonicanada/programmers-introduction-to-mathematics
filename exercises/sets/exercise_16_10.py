import random
import hashlib
import base64


def es_primo(n, k=5):
    # Función para verificar si un número es primo usando el algoritmo de Miller-Rabin
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Escribir n como (2^r) * d + 1
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    # Realizar el test de Miller-Rabin k veces
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(2, r, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generar_numero_primo_n_bits(n):
    while True:
        # Generar un número aleatorio de n dígitos
        numero = random.getrandbits(n)
        if es_primo(numero):
            return numero


def encriptar_mensaje(mensaje, public_key):
    e = public_key[0]
    n = public_key[1]
    c = mensaje**e % n
    return c


def desencriptar_mensaje(mensaje_cifrado, private_key):
    d = private_key[0]
    n = private_key[1]
    # m = mensaje_cifrado**d % n
    m = pow(mensaje_cifrado, d, n)
    print(m)
    return m


def calcula_mcd(a, b):
    # Calcula el Máximo Común Múltiplo
    while b != 0:
        a, b = b, a % b
    return a


def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, x, y = extended_gcd(b % a, a)
        return g, y - (b // a) * x, x


def modinv_manual(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('El inverso modular no existe')
    else:
        return x % m


def modinv(a, m):
    # Calcula el inverso modular de 'a' modulo 'm' utilizando pow
    return pow(a, -1, m)


def generar_claves(numbits_primo):
    # Se escojen 2 números primos
    p = generar_numero_primo_n_bits(numbits_primo)
    q = generar_numero_primo_n_bits(numbits_primo)
    n = p * q
    e = 65537  # Se usa normalmente este número
    public_key = [e, n]
    phi = (p - 1) * (q - 1)
    d = modinv(e, phi)
    private_key = [d, n]
    return public_key, private_key


# 1. Generamos claves
pub_k, pri_k = generar_claves(500)

# 2. Generar número secreto (mensaje)
mensaje = 12094883249819413298

# 3. Encriptar mensaje
secret = encriptar_mensaje(mensaje, pub_k)

# 4. Desencriptar mensaje
mensaje_desencriptado = desencriptar_mensaje(secret, pri_k)

# 5. Chequeo
print(secret)
print(mensaje == mensaje_desencriptado)
