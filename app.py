from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Carpeta donde se guardan las imágenes subidas y procesadas
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def area_double_integral(binary_mask):
    """
    Approximate area using double integral by summing pixel areas.
    Assumes each pixel area = 1 unit.
    """
    return np.sum(binary_mask > 0)

def area_polar_coordinates(radii, theta):
    """
    Calculate area using polar coordinates formula:
    A = ∫[θ1, θ2] ∫[0, R(θ)] r dr dθ
    radii: array of radius values as function of theta
    theta: array of theta values
    """
    # Numerical integration using trapezoidal rule
    # Integral over r: (1/2)*R(θ)^2
    r_squared = 0.5 * radii**2
    area = np.trapz(r_squared, theta)
    return area

def area_greens_theorem(contour):
    """
    Calculate area using Green's theorem:
    A = (1/2) ∮ (x dy - y dx)
    contour: Nx2 array of points
    """
    x = contour[:, 0]
    y = contour[:, 1]
    # Shifted arrays for vectorized calculation
    x_shift = np.roll(x, -1)
    y_shift = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * (y_shift - y) - y * (x_shift - x)))
    return area

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def home():
    image_path = None
    msg = ''
    area_double = None
    area_polar = None
    area_green = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Procesar la imagen (escala de grises)
            img = cv2.imread(filepath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Umbral para obtener máscara binaria
            _, binary_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

            # Calcular área usando doble integral (conteo de píxeles)
            area_double = area_double_integral(binary_mask)

            # Encontrar contornos para aplicar Green's theorem and polar coordinates
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Usar el contorno más grande para cálculo
                largest_contour = max(contours, key=cv2.contourArea)
                contour_points = largest_contour[:, 0, :]  # Nx2 array

                # Área con Green's theorem
                area_green = area_greens_theorem(contour_points)

                # Para área en coordenadas polares, convertir contorno a polar coords
                # Centrar contorno en su centroide
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0
                shifted_points = contour_points - np.array([cx, cy])

                # Convertir a coordenadas polares
                radii = np.sqrt(shifted_points[:, 0]**2 + shifted_points[:, 1]**2)
                theta = np.arctan2(shifted_points[:, 1], shifted_points[:, 0])

                # Ordenar por theta para integración
                sort_idx = np.argsort(theta)
                radii_sorted = radii[sort_idx]
                theta_sorted = theta[sort_idx]

                # Calcular área en coordenadas polares
                area_polar = area_polar_coordinates(radii_sorted, theta_sorted)

            # Guardar imagen procesada (binaria) para mostrar
            processed_filename = 'processed_' + filename
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, binary_mask)

            msg = '¡Imagen analizada y área calculada con éxito!'
            image_path = processed_filename  # Solo el nombre, no la ruta completa

    return render_template('index.html', msg=msg, image_path=image_path,
                           area_double=area_double, area_polar=area_polar, area_green=area_green)

# Ruta para mostrar imágenes de uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
