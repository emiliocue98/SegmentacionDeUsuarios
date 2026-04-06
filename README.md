# SegmentacionDeUsuarios
Actividad Machine Learning

**Integrantes del equipo:**
Gloria Janeth Esparza Martinez
Citlalli Izel Olmedo Paredes
Emilio Cue Funes

Trabajamos la tarea tambien en Colab porque teniamos problemas con el archivo que estaba muy pesado, por si hay algun problema aqui anexamos el link a la libreta https://colab.research.google.com/drive/10P9wcjQvAar0RfVqy01v9QtyVUESj3bd?usp=sharing#scrollTo=qLqltvwZIR_A

## Instrucciones para correr el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/emiliocue98/SegmentacionDeUsuarios.git
cd SegmentacionDeUsuarios
```

### 2. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset
```bash
python download_data.py
```
Esto descarga automáticamente el CSV desde Google Drive
y lo guarda localmente. Puede tardar unos minutos.

### 4. Verificar que el dataset se descargó
```bash
ls -lh product.csv
```

### 5. Abrir la libreta del integrante deseado
En la rama main encontrarás una libreta por integrante:

| Archivo | Integrante |
|---------|------------|
| libreta_gloria.ipynb | Gloria Esparza |
| libreta_citlalli.ipynb | Citlalli Olmedo |
| libreta_emilio.ipynb | Emilio Cue |

Abre el archivo y ejecuta todas las celdas en orden
de arriba hacia abajo.

- Celda por celda: `Ctrl + Enter`
- Todas de una vez: botón **Run All**

### 6. O puedes ver el trabajo por rama
```bash
# Ver todas las ramas
git branch -a

# Cambiarse a la rama deseada
git checkout Gloria-Esparza
git checkout Citlalli-Olmedo
git checkout EmilioCue
```

---

> ⚠️ **Notas importantes:**
> - El archivo CSV no está en el repositorio porque pesa
>   1.2 GB. Siempre descárgalo con `python download_data.py`
> - Cada libreta es independiente y contiene el trabajo
>   de cada integrante

**Actividad 1** 

**1.1 Análisis Exploratorio (EDA)**

1. Resumen general
Número total de eventos (interacciones): 8,471,220 (aunque el análisis actual se basa en un subconjunto de 200,000 interacciones debido a product_df = product1_df.head(200000).copy()). Para el product_df actual es 200,000.

Número de usuarios únicos: 76,119

Número de productos únicos: 125 (se calculará en la celda anterior).

2. Distribución de Eventos por Tipo (title)

La distribución de tipos de eventos (title) muestra las siguientes frecuencias:

banner_show: Mayoría de las interacciones, lo que sugiere que muchos usuarios ven banners sin hacer clic.

banner_click: Un número significativo de clics en banners.

order: Un menor número de eventos de compra, lo cual es esperable ya que la conversión suele ser menor que las interacciones de visualización o clic.

Otros tipos de eventos: También se observan otras interacciones como view, add_to_cart, etc., pero con menor frecuencia.

3. Distribución por Versión del Sitio (site_version)

La proporción entre desktop y mobile es aproximadamente 54.1% desktop y 45.9% mobile, lo que indica una base de usuarios considerable en ambas plataformas, con una ligera preferencia por desktop.

4. Productos Más Populares

Los 3 productos más populares son:

sneakers

sport_nutrition

company

5. Problemas Identificados

Valores Faltantes: No se encontraron valores faltantes en ninguna columna del product_df (No hay valores faltantes en el DataFrame.).

Anomalías: No se identificaron anomalías evidentes en los datos numéricos (target) ni en las distribuciones de las columnas categóricas durante el análisis exploratorio inicial.

**1.2 Ingeniería de características**

Durante la fase de ingeniería de características, se crearon diversas variables para capturar diferentes aspectos del comportamiento del usuario, con el objetivo de construir perfiles más ricos y discriminatorios para el clustering:

total_interactions: Representa el número total de eventos (interacciones) de un usuario en la plataforma. Es un indicador clave del nivel general de actividad y engagement del usuario.

total_clicks: Mide el número total de clics en banners (banner_click) realizados por un usuario. Esta característica ayuda a identificar usuarios más activos y propensos a explorar productos.

total_purchases: Indica el número total de órdenes (order) completadas por un usuario. Es una métrica directa de la propensión a la compra y el valor transaccional del usuario.

click_to_purchase_ratio: Esta proporción (total_clicks / total_purchases) mide la eficiencia con la que los clics de un usuario se traducen en compras. Un ratio bajo podría indicar que el usuario es indeciso o encuentra fricción en el proceso de compra, mientras que un ratio alto (donde total_purchases > 0) podría indicar un usuario que compra rápidamente después de un clic. Se maneja total_purchases = 0 para evitar divisiones por cero.

mobile_percentage: Porcentaje de eventos (total_events) que un usuario realiza desde la versión móvil del sitio. Esta característica ayuda a segmentar usuarios según su plataforma preferida de interacción.

desktop_percentage: Porcentaje de eventos (total_events) que un usuario realiza desde la versión de escritorio del sitio. Complementa la característica mobile_percentage para entender las preferencias de dispositivo del usuario.

avg_time_between_events_hours: El tiempo promedio (en horas) entre dos eventos consecutivos de un mismo usuario. Esta característica puede revelar patrones de uso, como usuarios que interactúan esporádicamente (largos intervalos) o aquellos que tienen sesiones de uso intensivo y continuado (cortos intervalos).

unique_products_viewed: Número de productos únicos en los que un usuario hizo clic (banner_click). Esta métrica refleja la diversidad de intereses del usuario o su tendencia a explorar un amplio catálogo.

unique_products_purchased: Número de productos únicos que un usuario compró (order). Similar a la anterior, pero enfocada en la diversidad de productos que finalmente adquiere el usuario.

show_to_click_ratio: Proporción de total_clicks sobre total_shows (banner_show). Mide la efectividad de los banners para generar clics después de ser mostrados. Un ratio alto indica mayor engagement con los banners.

days_active: Número de días distintos en los que un usuario ha interactuado con la plataforma. Esta característica es un indicador de la recurrencia y lealtad del usuario a lo largo del tiempo.

conversion_rate: Tasa de conversión de un usuario, calculada como total_purchases / total_interactions. Mide la efectividad general del usuario para realizar una compra en relación con todas sus interacciones.

Se utilizó el escalado de características (StandardScaler) antes de aplicar los algoritmos de clustering. Esta decisión se tomó por las siguientes razones:

Diferentes Escalas y Rangos: Las características creadas tienen rangos y escalas muy diferentes (e.g., total_interactions puede ser un número grande, mientras que click_to_purchase_ratio es una proporción). Algoritmos basados en distancias, como K-Means, son sensibles a estas diferencias, lo que podría llevar a que características con rangos más amplios dominen la función de distancia.

Importancia Equitativa: Al estandarizar las características para que tengan una media de 0 y una desviación estándar de 1, se asegura que todas las características contribuyan de manera equitativa al cálculo de la distancia entre los puntos, evitando sesgos hacia características con mayores valores numéricos.

Mejora del Rendimiento: La estandarización generalmente mejora el rendimiento y la convergencia de muchos algoritmos de aprendizaje automático, incluido K-Means.

Número Final de Características y Usuarios en la Matriz

Número final de características: 12 (excluyendo el user_id)

Número de usuarios: 76,119

La matriz utilizada para el clustering (X_scaled_df) tiene, por lo tanto, una forma de (76119, 12).

**Actividad 2** 

**2.1 Método del codo (elbow method)**

"El "codo" de la gráfica se localiza claramente en:
k=4 
En este valor, la curva experimenta su cambio de dirección más brusco antes de entrar en una fase de rendimientos decrecientes.
Valor de  k  Recomendado
Recomendación principal:  k=4 .
Es el punto de equilibrio óptimo entre la cohesión de los grupos y la simplicidad del modelo.

**2.2 Silhouette Score**

Silhouette Score: El Silhouette Score más alto se obtuvo con k = 8.

**2.3 Davies-Bouldin Index**

Davies-Bouldin Index: El Davies-Bouldin Index más bajo se obtuvo con k = 7.

**2.4 Decisión final sobre k**

Entre los tres, el Silhouette Score y el Davies-Bouldin Index son generalmente más confiables que el Método del Codo, ya que proporcionan una medida cuantitativa de la calidad del agrupamiento, reduciendo la subjetividad.
Considerando que el Silhouette Score sugiere k=8 y el Davies-Bouldin Index sugiere k=7, es común ver ligeras variaciones entre ellos.
El Davies-Bouldin Index de 7 es el valor más bajo, indicando la mejor separación de clústeres.
El valor de k que se usara en los algoritmos de clustering es: k = 7.

**Actividad 3** 
**3.1 Clustering con K-means**

**Parámetros usados en K-means**
Se aplicó el algoritmo K means utilizando un número de clusters previamente determinado, con el objetivo de segmentar los datos en grupos homogéneos. Se empleó el método de inicialización k means plus plus, el cual mejora la selección inicial de los centroides y favorece una mejor convergencia del modelo. Asimismo, se configuró un máximo de 300 iteraciones para asegurar que el algoritmo tuviera suficientes oportunidades de estabilizar los centroides. Se utilizaron múltiples inicializaciones mediante el parámetro n init, lo que permite ejecutar el algoritmo varias veces con diferentes puntos de partida y seleccionar la mejor solución en función de la menor inercia. Además, se estableció una semilla aleatoria mediante random state para garantizar la reproducibilidad de los resultados. Estos parámetros en conjunto permiten obtener una segmentación más robusta, evitando soluciones subóptimas y mejorando la calidad del agrupamiento.

La inercia final obtenida fue de 211253.46, lo cual representa la suma de las distancias cuadradas de cada punto a su centroide más cercano, es decir, el grado de compactación de los clusters. Este valor indica qué tan agrupados se encuentran los datos dentro de cada cluster; a menor inercia, mayor cohesión interna. En este caso, la inercia refleja un nivel adecuado de agrupamiento considerando la complejidad y dimensionalidad del conjunto de datos, aunque su interpretación debe complementarse con otras métricas como el silhouette score y el índice de Davies Bouldin para evaluar de manera más integral la calidad del modelo.

**Tabla de centroides**
total_interactions	total_clicks	total_purchases	click_to_purchase_ratio	mobile_percentage	desktop_percentage	avg_time_between_events_hours	unique_products_viewed	unique_products_purchased	show_to_click_ratio	days_active	conversion_rate
Cluster 0	-0.059604	-0.440829	-0.265914	-0.173023	-0.226885	0.226885	3.988241	-0.454685	-0.270591	-0.395368	0.087195	-0.246684
Cluster 1	0.611676	-0.437125	3.405780	-0.137546	-0.906959	0.906959	0.475173	-0.450628	3.471594	-0.389022	0.838634	4.024631
Cluster 2	0.234288	1.506724	-0.265914	-0.173023	0.390879	-0.390879	-0.104441	1.628751	-0.270591	2.053028	-0.112323	-0.246684
Cluster 3	-0.406645	-0.470194	-0.265914	-0.173023	-1.462951	1.462951	-0.209299	-0.486851	-0.270591	-0.425975	-0.342576	-0.246684
Cluster 4	-0.327495	-0.470293	-0.265914	-0.173023	0.681669	-0.681669	-0.224347	-0.486959	-0.270591	-0.425975	-0.256087	-0.246684
Cluster 5	1.746904	1.905778	3.488204	4.717300	0.052112	-0.052112	0.093234	1.912020	3.543276	1.351284	1.509180	2.340884
Cluster 6	2.667764	1.959915	-0.262472	-0.173023	0.528634	-0.528634	0.085294	1.897387	-0.266991	0.310076	2.495427	-0.245790

El análisis de los centroides obtenidos mediante el algoritmo K means permitió identificar distintos perfiles de comportamiento entre los usuarios. Se observa que los clusters 5 y 1 representan a los usuarios de mayor valor, destacando especialmente el cluster 5 por sus altos niveles de interacción, clics, compras y conversión, lo que indica un alto grado de compromiso con la plataforma. Por otro lado, los clusters 2 y 6 agrupan usuarios con alta actividad reflejada en un elevado número de clics, productos vistos y días activos pero con baja conversión, lo que sugiere oportunidades de mejora en estrategias de persuasión o experiencia de usuario. En contraste, los clusters 0, 3 y 4 corresponden a usuarios con baja interacción, escaso nivel de compras y menor participación general, considerados como segmentos pasivos o de bajo impacto. Los resultados evidencian una segmentación clara que permite identificar tanto a los usuarios más rentables como a aquellos que requieren estrategias específicas para incrementar su nivel de engagement y conversión.

**Métricas de calidad**
Métricas de calidad (silhouette, intra/inter-cluster distance, Davies-Bouldin)

Silhouette Score: 0.619851679518303
Intra-cluster distance (Inercia): 211253.45727269168
Distancias entre centroides: [7.74678423 5.63629339 4.58072435 4.42597891 9.69642504 6.43476665
 8.03418393 7.01807925 7.30422636 6.65133933 8.28642734 4.6773507
 3.87943881 8.0726424  4.01392308 3.03525335 9.30362016 6.13658535
 9.06212159 5.3682342  7.90910696]
Davies-Bouldin Index: 0.8828982121995878

**Visualización de clusters**
Los clusters presentan en general una buena separación en el espacio reducido mediante PCA, lo que indica que el modelo K means logró identificar grupos con características diferenciadas. Se observan clusters claramente definidos, particularmente en las regiones derecha inferior del gráfico, donde la distancia entre grupos es mayor. No obstante, existe cierto grado de solapamiento en la zona izquierda, lo que sugiere que algunos usuarios comparten características similares y podrían no estar completamente diferenciados. En conjunto, la segmentación es adecuada, aunque con áreas donde la separación entre clusters podría mejorarse.

**Tabla de distribución de usuarios por cluster**
Número de usuarios	Porcentaje (%)
0	2747	3.608823
1	2776	3.646921
2	9904	13.011206
3	18563	24.386815
4	35468	46.595462
5	2673	3.511607
6	3988	5.239165

**Parámetro de bandwidth usado y justificación**
Debido al tamaño del dataset (200,000 observaciones), se utilizó una muestra aleatoria de 10,000 datos para estimar el parámetro bandwidth, reduciendo el costo computacional sin afectar significativamente la representatividad de los resultados.

El parámetro bandwidth fue estimado utilizando la función estimate bandwidth con distintos valores de quantile (0.2, 0.3 y 0.4). Se seleccionó el valor correspondiente a quantile = 0.3, obteniendo un bandwidth de aproximadamente 2.94, ya que proporciona un equilibrio adecuado entre la cantidad de clusters generados y la separación entre los mismos, evitando tanto la sobresegmentación como la generalización excesiva de los datos.

**Número de clusters encontrados automáticamente**
Según la muestra, el algoritmo Mean Shift identificó un total de 36 clusters, lo cual es significativamente mayor en comparación con los 7 clusters obtenidos mediante K means. Esto indica que Mean Shift es más sensible a la densidad de los datos y tiende a generar una segmentación más detallada, identificando subgrupos dentro de los clusters principales.

**Métricas de calidad**
Se obtuvo un silhouette score de 0.46, lo que indica una separación moderada entre los clusters, inferior a la obtenida con K means, sugiriendo una menor claridad en la delimitación de los grupos. La distancia intra cluster fue de 44502.21, reflejando una buena compactación interna considerando el alto número de clusters generados. Por otro lado, las distancias inter cluster presentan una amplia variabilidad, con valores que van desde distancias pequeñas hasta separaciones considerablemente grandes, lo que indica que algunos clusters están bien diferenciados mientras que otros se encuentran más cercanos entre sí. El índice de Davies Bouldin de 0.71, al ser menor a 1, indica un buen desempeño del modelo en términos de cohesión y separación.

**Comparativa inicial: K-means vs Mean-Shift**
Mean Shift encontró más clusters que el valor de k seleccionado en K means, ya que identificó 36 clusters frente a los 7 definidos previamente. Esto evidencia que Mean Shift es más sensible a la densidad de los datos y tiende a generar una segmentación más detallada, detectando subgrupos dentro de los clusters principales.
La distribución de usuarios por cluster obtenida mediante Mean Shift evidencia una alta concentración en pocos grupos principales, destacando el cluster 0 que agrupa el 74% del total.
En cuanto a la comparación observada por las gráficas, los clusters obtenidos son más fragmentados y menos homogéneos que los de K means, presentando formas irregulares y una mayor dispersión. Aunque algunas regiones del espacio coinciden con las agrupaciones generales identificadas por K means, en general los resultados son más granulares y diferentes, lo que podría ofrecer mayor detalle pero dificulta su interpretación y aplicación práctica.

**3.3 Clustering con DBSCAN**
**K-distance graph con eps elegido**
En este caso, el codo se ubicó aproximadamente en un valor de 0.4, por lo que se seleccionó eps = 0.4 como el parámetro óptimo. Este valor permite un adecuado equilibrio entre la formación de clusters y la detección de ruido, evitando tanto la sobreagrupación como la fragmentación excesiva de los datos.

**Parámetros usados**
eps=0.4

Número de clusters encontrados
**% de usuarios clasificados como ruido**
Los usuarios clasificados como ruido por DBSCAN corresponden principalmente a datos atípicos, ya que presentan comportamientos distintos al resto y no se agrupan en regiones de alta densidad. Estos usuarios se caracterizan por patrones irregulares o extremos en sus interacciones, lo que los separa de los clusters principales. Aunque es más probable que representen anomalías dentro del dataset, no se descarta que algunos puedan corresponder a bots, usuarios nuevos o comportamientos poco comunes.

**Actividad 4** 
**4.1 - Matriz Comparativa**

     Métrica K-Means Mean-Shift  DBSCAN
       Número de clusters       7         38     306
         Silhouette Score  0.6199     0.4555   0.474
     Davies-Bouldin Index  0.8829     0.5943   0.951
   Intra-cluster distance  1.2023     1.8252  0.3804
   Inter-cluster distance  6.5372    11.0228  7.0451
    % usuarios como ruido      0%         0%   2.12%
Tiempo de ejecución (seg)    1.93      66.21   55.25
        Interpretabilidad    Alta       Baja    Baja


**4.2 - Recomendación de algoritmo**

Métricas de calidad

Revisando las métricas de calidad que se calcularon, tenemos que en Silhouette Score el valor mayor lo obtuvo Kmeans con 0.6199 por lo cual en esta métrica es el mejor.

En cuanto a Davies-Bouldin Index el valor menor es el mejor y en este caso lo obtuvo Mean-Shiftncon 0.6071.

En la distancia de Intra-cluster tenemos que el menor valor esta en DBSCAN con 0.3804.

Para la distancia Inter-cluster el valor mayor lo obtuvo Mean-Shift con 10.1007.

En cuanto a el tiempo de ejecución menor, el mejor tiempo lo obtuvo Kmeans con 2.06 segurndos.

K-Means obtuvo el mejor Silhouette Score, que es la métrica más importante ya que mide que tan bien estan separados los clusters y que estos son sohesivos, que quiere decir que los datos que forman parte de un mismo cluster son muy similares entre sí, están estrechamente relacionados.

Mean-Shift gana en Davies-Bouldin y en la distancia Inter-cluster, sin embargo, estos valores se explican por la fragmentación tan grande al generar 44 clusters muy pequeños y dispersos, los centroides naturalmente quedan más alejados entre sí y los grupos internamente más compactos, pero eso no refleja una segmentación que sea útil para el negocio.

DBSCAN obtiene una buena distancia Intra-cluster porque crea cientos de grupos diminutos que son muy compactos, pero que en la práctica no sirven porque no puedes operar 306 estrategias de marketing distintas.

En cuanto al tiempo de ejecución, Kmeans es el más rápido con 2.06 segundos, en comparacion con Mean-Shift con 54.21 segundos y DBSCAN con 52.10 segundos.

Interpretabilidad de los resultados

K-Means generó 7 segmentos cada uno con un perfil de comportamiento distinto que el equipo de marketing puede entender y facilita la toma de desiciones.

Mean-Shift generó 44 clusters y DBSCAN generó 306, lo que hace dificil diseñar tantas estrategías de marketingy no es viables para la toma de decisiones de negocio.

Manejo de outliers

DBSCAN clasificó el 2.12% de usuarios como ruido y los dejó sin ningún segmento asignado, fuera del análisis. En un problema de negocio esto representa clientes ignorados y oportunidades comerciales perdidas.

K-Means y Mean-Shift clasifican al 100% de los usuarios, pero solo K-Means lo hace en un número de clusters razonable.

Aplicabilidad al problema empresarial

El gerente necesita decidir entre vender el espaciode banners bajo un modelo CPC o invertir en personalización por segmento de usuario.

K-Means entrega exactamente lo que se necesita: 7 perfiles de usuario accionables con características claras de comportamiento de compra, uso de dispositivo y nivel de engagement. Esta segmentación permite personalizar el contenido de cada banner según el perfil del usuario, maximizando el retorno de inversión de cada impresión mostrada. Además de que K-Means tardó solo 2.06 segundos en ejecutarse, frente a los 54.21 de Mean-Shift y 52.10 de DBSCAN, lo que lo hace también el más eficiente.

¿Porqué no Mean Shift? Generó 44 clusters sin utilidad operacional.En el métrico de Silhouette obtuvo 0.4577, inferior al de K-Means 0.6199, además de que su tiempo de ejecución de 54.21 segundos, el más lento.

¿Porqué no DBSCAN? Generó 306 clusters: completamente inoperable, además dejó el 2.12% de usuarios sin clasificar como ruido. En el métrico de Silhouette obtuvo 0.4740, inferior al de K-Means 0.6199. En cuanto a Davies-Bouldin obtuvo 0.9510, el peor de los tres.

K-Means con k=7 es el algoritmo recomendado porque obtuvo el mejor Silhouette Score (0.6199), clasifica al 100% los usuarios, produce 7 segmentos para el equipo de marketing y es el más rápido con solo 2.06 segundos de ejecución.        

**5.1 Perfiles detallados de clústeres**


Interpretación de los 7 Clústeres:

**Clúster 0** — Visitantes Mobile Inactivos (2,747 usuarios | 3.6%)
0 compras, 0 conversión, casi 0 clics (0.016)
Entran desde mobile (57.5%), ven el sitio y se van sin interactuar.
Son usuarios que probablemente llegaron por un anuncio o enlace pero no encontraron lo que buscaban.
Valor comercial: Muy bajo.


**Clúster 1** — Compradores Desktop (2,776 usuarios | 3.6%)
1.07 compras promedio, tasa de conversión del 32.5% — el segundo grupo más valioso.
Casi no hacen clics (0.018) pero compran directamente, lo que indica que llegan con intención de compra clara.
Prefieren desktop (solo 25.9% mobile) y tienen 3.9 días activos.
Valor comercial: Muy alto. Saben lo que quieren y compran sin dudar.


**Clúster 2** — Exploradores Mobile (9,904 usuarios | 13.0%)
0 compras a pesar de 1.07 clics promedio — exploran bastante pero no convierten.
Son mayormente mobile (86.2%) con 2.1 días activos.
Hacen clic en productos pero algo los detiene antes de comprar: precio, proceso de pago, falta de confianza.
Valor comercial: Medio. Gran potencial de conversión si se elimina la fricción.


**Clúster 3** — Pasivos Desktop (18,563 usuarios | 24.4%)
0 compras, 0 clics, 0 conversión. Prácticamente no usan mobile (0.1% mobile = casi 100% desktop).
Son el segundo grupo más grande. Entran al sitio, ven banners pero no interactúan con nada.
Posiblemente usuarios que llegan por error, robots de indexación, o personas que no son el público objetivo.
Valor comercial: Muy bajo.


**Clúster 4** — Pasivos Mobile (35,468 usuarios | 46.6%)
El grupo más grande del dataset — casi la mitad de todos los usuarios.
0 compras, 0 clics, 0 conversión. Usan mobile casi exclusivamente (99.7%).
Entran, ven la página y se van. Muy baja intención de compra.
Valor comercial: Muy bajo. Sin embargo, por su volumen enorme, cualquier pequeña mejora en engagement impacta significativamente.


**Clúster 5** — Compradores Mobile Activos (2,673 usuarios | 3.5%)
El segmento más valioso del análisis. 1.09 compras promedio, conversión del 19.7%, 1.29 clics y 5.3 días activos.
Son mobile (70.5%), muy comprometidos con el sitio y vuelven frecuentemente.
Combinan exploración activa con compra efectiva — son el cliente ideal.
Valor comercial: Máximo. Prioridad absoluta de retención.


**Clúster 6** — Hipernavegantes Sin Conversión (3,988 usuarios | 5.2%)
Los usuarios más recurrentes (7.2 días activos) pero con 0 compras.
Tienen el mayor número de clics (1.32) y son casi exclusivamente mobile (92.6%).
Vuelven una y otra vez, exploran mucho, pero nunca compran. Son indecisos crónicos o usuarios que usan el sitio como catálogo pero compran en otro lugar.
Valor comercial: Medio-alto potencial. Si se logra convertirlos, el impacto sería significativo dado su alta recurrencia.
