# Análisis de Sentimiento con MLflow y Control de Versiones

## 🎯 Objetivo del Proyecto

Este proyecto tiene como objetivo construir un pipeline reproducible de entrenamiento de un modelo de análisis de sentimiento utilizando `MLflow` para el control de experimentos y versiones. Se aplican prácticas de MLOps para monitorizar, versionar y documentar todo el proceso de Machine Learning.

---

## 📊 Dataset Utilizado

Se ha utilizado un subconjunto del dataset de tweets etiquetados disponible públicamente:

- URL del dataset:  
  [https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv)
  
Este dataset contiene tweets clasificados en tres categorías: **positivos, negativos y neutrales**.

---

## 🔁 Flujo del Pipeline

1. **Carga del dataset**
2. **Preprocesamiento de texto** usando `TfidfVectorizer`
3. **División del dataset** en entrenamiento y test (80/20)
4. **Entrenamiento del modelo** con `RandomForestClassifier`
5. **Evaluación con accuracy**
6. **Registro del experimento y métricas** con MLflow
7. **Log del modelo entrenado**
8. **Registro del modelo con control de versiones**
9. **Exposición de la interfaz web de MLflow** vía `pyngrok`

---

## 📦 Librerías Necesarias

Asegúrate de tener las siguientes librerías instaladas:

```bash
pip install pandas scikit-learn mlflow pyngrok
