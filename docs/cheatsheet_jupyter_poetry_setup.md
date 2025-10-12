
# ✅ Cheatsheet — Set Up de Jupyter con Poetry para Proyectos MLOps

## 📁 Estructura recomendada de carpetas

```
mlops_project/
├── data/
├── docs/
├── models/
├── notebooks/
├── references/
├── reports/
├── src/
├── tests/
├── pyproject.toml
├── README.md
```

---

## 1️⃣ Inicializar un nuevo proyecto con Poetry

```bash
poetry init
# o, si quieres crear todo automáticamente:
poetry init --name mlops-project --dependency ""
```

---

## 2️⃣ Instalar `notebook` como dependencia de desarrollo

```bash
poetry add --group dev notebook
```

🔎 Esto añadirá en `pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
notebook = "^7.3.3"
```

---

## 3️⃣ Instalar `pandas` como dependencia principal

```bash
poetry add pandas
```

🔎 Esto añadirá en `pyproject.toml`:

```toml
[project]
dependencies = [
    "pandas^2.2.3"
]
```

---

## 4️⃣ Activar el entorno virtual

Ver ruta del entorno:

```bash
poetry env info --path
```

Ejemplo de salida:
```
D:\Workspace\mlops_project\.venv
```

Activar en PowerShell:

```powershell
& "D:\Workspace\mlops_project\.venv\Scripts\Activate.ps1"
```

---

## 5️⃣ Iniciar Jupyter Notebook

Una vez en el entorno virtual:

```bash
jupyter notebook
```

Esto abrirá `http://localhost:8888/tree` en el navegador.

---

## ✅ Verificación rápida en el notebook

```python
import pandas as pd
print(pd.__version__)
```

---

## 🧹 Consejo adicional

Agrega `.venv` al `.gitignore` si usas Git:

```
# .gitignore
.venv/
```
