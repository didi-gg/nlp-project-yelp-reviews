# Análisis NLP de reseñas de Yelp

Proyecto de Procesamiento del Lenguaje Natural aplicado a reseñas de restaurantes de Yelp. El objetivo es estudiar la relación entre el lenguaje utilizado en una reseña y su valoración en estrellas, y construir un clasificador capaz de predecir ratings de 1 a 5 estrellas a partir de texto nuevo.

El proyecto combina análisis exploratorio, limpieza de texto, representaciones léxicas y semánticas, análisis de sentimiento, similitud entre reseñas, modelado de tópicos, extracción de aspectos y evaluación de modelos de aprendizaje automático.

## Preguntas del proyecto

- ¿Qué características del texto aparecen asociadas a ratings altos o bajos?
- ¿Cómo cambia el análisis cuando se utilizan palabras, bigramas, caracteres o embeddings?
- ¿Puede un modelo distinguir las cinco clases de rating, pese al desbalance del conjunto de datos?
- ¿Qué aspectos de la experiencia del cliente, como comida, servicio, precio, espera o ambiente, se mencionan con mayor frecuencia?
- ¿Cómo se comporta el modelo ante una reseña nueva y qué nivel de incertidumbre muestra?

## Flujo de trabajo

```text
Datos originales
	|
	v
Limpieza y exploración
	|
	v
Texto normalizado + señales expresivas + sentimiento
	|
	+--> BoW, TF-IDF, n-gramas y similitud
	+--> Embeddings de oración
	+--> Tópicos NMF y aspectos de negocio
	|
	v
Split estratificado de entrenamiento y prueba
	|
	v
Comparación de VADER, TF-IDF y embeddings
	|
	v
Modelo guardado y demo interactiva
```

## Notebooks

### 1. Limpieza y exploración de los datos

Carga los CSV y revisa la estructura, calidad y distribución de las reseñas. Esta etapa prepara el corpus limpio que utilizan los análisis posteriores.

### 2. NLP

Realiza el análisis lingüístico principal:

- conserva señales del texto original, como mayúsculas, puntuación, repeticiones y emojis;
- normaliza el texto para las representaciones léxicas;
- tokeniza y filtra stopwords, conservando negadores e intensificadores;
- construye representaciones BoW, TF-IDF, unigramas, bigramas y n-gramas de caracteres;
- genera embeddings con `sentence-transformers`;
- calcula sentimiento con VADER sobre el texto original;
- busca reseñas similares y posibles near-duplicates;
- descubre ocho tópicos con NMF;
- analiza los aspectos de comida, servicio, precio, espera y ambiente.

El análisis es exploratorio. Una asociación entre una palabra, un tópico o un aspecto y el rating no demuestra causalidad.

### 3. Evaluación de modelos ML

Convierte `Rating` en un problema de clasificación multiclase con cinco categorías. Usa un split estratificado de 80 % para entrenamiento y 20 % para prueba, manteniendo la misma distribución de estrellas en ambos conjuntos.

Se comparan cuatro referencias:

1. **VADER bucketizado:** línea base no entrenada. Sus cortes son heurísticos y sirven únicamente como referencia orientativa.
2. **TF-IDF con unigramas + Regresión Logística:** aprende asociaciones entre palabras y ratings.
3. **TF-IDF con unigramas y bigramas + Regresión Logística:** añade contexto local y obtiene el mejor rendimiento global del experimento.
4. **Embeddings de oración + Regresión Logística:** utiliza una representación semántica densa y obtiene el mejor recall para ratings bajos.

Como el target está desbalanceado, se utiliza `class_weight="balanced"`. La métrica principal es `Macro-F1`, complementada con `Balanced Accuracy`, recall de 1-2 estrellas, métricas por clase y matrices de confusión.

### 4. Demo

Carga el modelo guardado y permite introducir una reseña en inglés mediante un widget de `ipywidgets`. La demo muestra:

- el texto original;
- el texto normalizado que recibe el vectorizador;
- el rating predicho;
- las probabilidades estimadas para las cinco estrellas.

Las probabilidades ayudan a detectar predicciones ambiguas, pero no deben interpretarse como certezas ni como porcentajes perfectamente calibrados.

## Resultados principales

La evaluación disponible en el notebook muestra los siguientes valores sobre el conjunto de prueba:

| Modelo | Macro-F1 | Balanced Accuracy | Recall 1-2 estrellas |
| --- | ---: | ---: | ---: |
| VADER bucketizado | 0,335 | 0,323 | 0,431 |
| TF-IDF unigramas + Regresión Logística | 0,519 | 0,546 | 0,786 |
| TF-IDF unigramas y bigramas + Regresión Logística | **0,539** | **0,556** | 0,762 |
| Embeddings + Regresión Logística | 0,473 | 0,507 | **0,816** |

El modelo con TF-IDF y bigramas es la mejor opción cuando se prioriza el equilibrio entre las cinco clases. Los embeddings son preferibles si el objetivo principal es recuperar la mayor cantidad posible de reseñas de 1-2 estrellas. VADER cumple su función como línea base, pero no como solución final de clasificación.

## Datos y artefactos

Los archivos se encuentran en `data/`:

| Archivo | Descripción |
| --- | --- |
| `Yelp Restaurant Reviews.csv` | Datos originales. |
| `Yelp Restaurant Reviews_clean.csv` | Datos tras la limpieza inicial. |
| `yelp_enriquecido.parquet` | Corpus enriquecido con variables utilizadas en la evaluación. |
| `embeddings_oracion.npy` | Embeddings de oración de 384 dimensiones. |
| `train_split.csv` | Partición estratificada de entrenamiento. |
| `test_split.csv` | Partición estratificada de prueba. |
| `X_emb_train.npy` | Embeddings correspondientes al entrenamiento. |
| `X_emb_test.npy` | Embeddings correspondientes a la prueba. |
| `mejor_modelo.joblib` | Pipeline del modelo utilizado por la demo. |

El corpus utilizado en la evaluación contiene 19.894 reseñas. La clase mayoritaria y la minoritaria presentan una razón aproximada de 8,9x, por lo que las métricas deben interpretarse teniendo en cuenta el desequilibrio.

## Instalación

Se recomienda utilizar un entorno virtual o conda y Python 3.10 o superior.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn vaderSentiment \
    sentence-transformers joblib pyarrow ipywidgets jupyter
```

## Ejecución

1. Abre el proyecto en VS Code con la extensión de Jupyter instalada.
2. Selecciona un intérprete de Python con las dependencias anteriores.
3. Ejecuta los notebooks en orden:
   - `1.Limpieza y exploración de los datos.ipynb`
   - `2. NLP.ipynb`
   - `3. Evaluación de modelos ML.ipynb`
   - `4. Demo.ipynb`
4. Para la demo, asegúrate de que `data/mejor_modelo.joblib` exista y habilita los widgets de Jupyter.

La generación de embeddings puede descargar pesos de Hugging Face y requerir conexión a internet la primera vez. Si los embeddings ya están guardados en `data/`, pueden reutilizarse sin repetir esa etapa.

## Limitaciones y próximos pasos

- Las reseñas y los ratings pertenecen a un corpus concreto; el rendimiento puede cambiar en otros restaurantes, fechas o dominios.
- El desbalance hace especialmente importante revisar las métricas por clase y no solo una métrica global.
- Los errores se concentran previsiblemente entre ratings vecinos, que pueden utilizar un vocabulario muy parecido.

Como siguientes pasos se propone realizar validación cruzada estratificada, revisar los errores más frecuentes, calibrar las probabilidades y evaluar splits por fecha o restaurante. Después de estas comprobaciones se pude evaluar si tiene sentido medir si un transformer aporta una mejora suficiente frente al modelo TF-IDF con bigramas.
