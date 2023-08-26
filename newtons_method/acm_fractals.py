import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(z):
    return z**3 - 1

def df(z):
    return 3*z**2

# Newton-Raphson iteration
def newton_raphson(z, max_iters=100, tol=1e-6):
    for i in range(max_iters):
        z_new = z - f(z)/df(z)
        if abs(z_new - z) < tol:
            break
        z = z_new
    return z, i

def plot_newton_fractal_with_roots(width, height, x_center, y_center, x_width, y_width, max_iters=30):
    # Define the image and the domain
    img = np.zeros((width, height, 3), dtype=np.uint8)
    x_min, x_max = x_center - x_width/2, x_center + x_width/2
    y_min, y_max = y_center - y_width/2, y_center + y_width/2
    roots = np.array([1, -0.5 + 0.5j*np.sqrt(3), -0.5 - 0.5j*np.sqrt(3)])

    for x in range(width):
        for y in range(height):
            zx, zy = x * (x_max - x_min) / (width - 1) + x_min, y * (y_max - y_min) / (height - 1) + y_min
            z, i = newton_raphson(zx + zy*1j, max_iters)
            
            # Map the resulting root to a color with enhanced contrast using logarithmic scale
            color_intensity = int(255 * np.log(1 + i) / np.log(1 + max_iters))
            root_num = np.argmin(np.abs(roots - z))
            if root_num == 0:
                img[y, x, :] = [255 - color_intensity, 255 - color_intensity, 255]
            elif root_num == 1:
                img[y, x, :] = [255, 255 - color_intensity, 255 - color_intensity]
            elif root_num == 2:
                img[y, x, :] = [255 - color_intensity, 255, 255 - color_intensity]

    # Mark the roots with black dots
    for root in roots:
        x_root = int((root.real - x_min) / (x_max - x_min) * width)
        y_root = int((root.imag - y_min) / (y_max - y_min) * height)
        img[y_root-3:y_root+3, x_root-3:x_root+3, :] = [0, 0, 0]  # A small black square around the root
            
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# To plot the enhanced fractal with roots:
plot_newton_fractal_with_roots(800, 800, 0, 0, 2, 2)
