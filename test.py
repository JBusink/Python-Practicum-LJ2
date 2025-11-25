import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ipywidgets as widgets
from IPython.display import display

# Parameters
N = 256  # Aantal gridpunten
L = 10.0  # Ruimtelijke omvang
dx = L / N  # Gridstap
x = np.linspace(-L/2, L/2, N)

hbar = 1.0  # Gereduceerde Planck-constante
m = 1.0  # Massa van het deeltje
dt = 0.01  # Tijdstap
t_max = 2.0  # Maximale tijd

def initialize_simulation(V0, width, k0):
    """Initialiseert het golfpakket en de potentiaalbarrière."""
    global psi, V, T, V_prop
    
    # Potentiaalbarrière
    V = np.zeros(N)
    V[(x > -width/2) & (x < width/2)] = V0
    
    # Initiële Gaussiaanse golfpakket
    x0 = -3.0  # Startpositie
    sigma = 0.5  # Breedte van het golfpakket
    psi = np.exp(1j * k0 * x) * np.exp(-(x - x0)**2 / (2 * sigma**2))
    psi /= np.linalg.norm(psi)  # Normalisatie
    
    # Fourier ruimte voor de kinetische operator
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    T = np.exp(-1j * (hbar * kx**2 / (2 * m)) * dt)  # Kinetische propagator
    
    # Potentiaal propagator
    V_prop = np.exp(-1j * V * dt / hbar)

def time_step(psi):
    """Voert een tijdstap uit met de split-operator methode."""
    psi = np.fft.ifft(T * np.fft.fft(V_prop * psi))
    return psi

def calculate_coefficients():
    """Berekent reflectie- en transmissiecoëfficiënten."""
    global psi
    P_transmitted = np.sum(np.abs(psi[x > 2])**2)  # Kans voorbij de barrière
    P_reflected = np.sum(np.abs(psi[x < -2])**2)   # Kans vóór de barrière
    return P_reflected, P_transmitted

def update_plot(frame):
    """Werkt de plot bij tijdens de animatie."""
    global psi
    for _ in range(10):
        psi = time_step(psi)
    line.set_ydata(np.abs(psi)**2)
    R, T = calculate_coefficients()
    reflection_text.set_text(f"R = {R:.2f}")
    transmission_text.set_text(f"T = {T:.2f}")
    return line, reflection_text, transmission_text

def run_simulation(V0, width, k0):
    """Start de simulatie met de opgegeven parameters."""
    initialize_simulation(V0, width, k0)
    global fig, ax, line, reflection_text, transmission_text
    
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.abs(psi)**2, label="Kansdichtheid")
    ax.plot(x, V / V0 * np.max(np.abs(psi)**2), 'r--', label="Potentiaal")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Positie x")
    ax.set_ylabel("Kansdichtheid")
    ax.legend()
    
    reflection_text = ax.text(-4, 0.8, "R = 0.00", fontsize=12, color='blue')
    transmission_text = ax.text(-4, 0.7, "T = 0.00", fontsize=12, color='green')
    
    ani = animation.FuncAnimation(fig, update_plot, frames=int(t_max / (10 * dt)), interval=50)
    plt.show()

# Interactieve sliders
    widgets.interact(run_simulation, 
                 V0=widgets.FloatSlider(min=0, max=10, step=0.5, value=5, description="Barrièrehoogte V0"),
                 width=widgets.FloatSlider(min=0.5, max=3, step=0.1, value=1, description="Barrièrebreedte"),
                 k0=widgets.FloatSlider(min=1, max=10, step=0.5, value=5, description="Golftal k0"));


run_simulation(2,0.1,0.01)