# SegmentacionDeUsuarios
Actividad Machine Learning



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
