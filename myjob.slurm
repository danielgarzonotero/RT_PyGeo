#!/bin/bash
#SBATCH --nodes=1                  # Número de nodos
#SBATCH --ntasks=30               # Número de tareas (cores)
#SBATCH --time=72:00:00           # Duración máxima del trabajo 
#SBATCH --partition=gpu            # Partición que utiliza GPU
#SBATCH --gres=gpu:1               # Número de GPUs solicitadas
#SBATCH --account=bilodeaugroup    # Asignación/cuenta a utilizar

# Cargar los módulos necesarios
module purge
module load anaconda

source activate deep_rdkit_py37    # Activar el entorno virtual

# Navegar al directorio del proyecto
cd /home/vvd9fd/CSS_PyGeo

# Ejecutar el script de Python
python main.py
