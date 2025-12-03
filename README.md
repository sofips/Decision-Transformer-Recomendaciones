## Decision Transformer para Sistemas de Recomendación

#### Trabajo realizado por Sofía Perón y Felipe Ávila para la materia Aprendizaje por Refuerzos en el marco de la Diplomatura en Ciencia de Datos (2025)

Este proyecto implementa un sistema de recomendación basado en Decision Transformers (DTs) aplicado a un entorno offline de calificaciones de películas. El objetivo es evaluar si un modelo tipo Transformer, originalmente diseñado para modelar secuencias en NLP y posteriormente extendido al aprendizaje por refuerzo (RL), puede capturar patrones usuario–ítem de forma competitiva frente a métodos tradicionales de recomendación.

La idea se basa en interpretar una sesión de interacción usuario–sistema como una trayectoria de un MDP y entrenar el DT para imitar comportamientos asociados a retornos altos. Además, el repositorio incluye una comparación con un código de referencia y versiones modificadas que introducen mejoras en embeddings, organización de secuencias y arquitectura de la cabeza de predicción.

### 📁 Estructura del repositorio
```text
Decision-Transformer-Recomendaciones/
├── data
│   ├── groups
│   ├── processed
│   ├── test_users
│   └── train
├── models
├── notebooks
│   └── checkpoints
├── reference_code
│   └── checkpoints
├── results
│   ├── trained_models
│   └── training_histories
├── scripts
└── src
    ├── data
    ├── evaluation
    ├── models
    └── training
```

### 🚀 Instalación

El proyecto utiliza las siguientes librerías :

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Matplotlib



El entorno completo puede instalarse utilizando el archivo `requirements.yml`.


```bash
conda env create -f requirements.yml
```
