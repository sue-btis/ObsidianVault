#Python #Matplotlib

[[2 Basic Plot]]

# Resumen de Matplotlib y Seaborn

## Introducción a Matplotlib

Matplotlib es la biblioteca fundamental de visualización de gráficos en Python. Es ampliamente utilizada para crear gráficos 2D y proporciona una gran flexibilidad para personalizar cada aspecto del gráfico.

### Características Principales:

- **Versatilidad:** Permite crear gráficos como líneas, barras, dispersión, histogramas, y gráficos de torta.
    
- **Alta Personalización:** Permite modificar colores, etiquetas, títulos, ejes, cuadrículas, estilos de línea, y más.
    
- **Gráficos Complejos:** Soporta subplots y gráficos en cuadrícula mediante `plt.subplots()`.
    
- **Compatibilidad:** Trabaja bien con librerías como **NumPy** y **Pandas**.
    
- **APIs Flexibles:** Soporta tanto la API funcional (`plt.plot()`) como la orientada a objetos.
    
### Ventajas y Desventajas de Matplotlib:

- ✅ Gran flexibilidad y personalización.
    
- ✅ Compatible con otras bibliotecas científicas.
    
- ❌ Requiere mucho código para gráficos complejos.
    
- ❌ Estilo predeterminado algo básico.
    

##  Seaborn

### Ventajas y Desventajas de Seaborn:

- ✅ Gráficos complejos y visualmente atractivos.
    
- ✅ Integración directa con Pandas.
    
- ❌ Menos control que Matplotlib para gráficos personalizados.
    
- ❌ Dependencia de Matplotlib como backend gráfico.
    

## Comparación: Matplotlib vs Seaborn

|Aspecto|Matplotlib|Seaborn|
|---|---|---|
|Flexibilidad|Muy alta, pero requiere mucho código.|Baja, pero muy sencillo de usar.|
|Estética|Básica, necesita personalización manual.|Atractiva por defecto.|
|Integración con Pandas|Limitada.|Alta, directo desde DataFrames.|
|Análisis Estadístico|No directamente soportado.|Incluye gráficos estadísticos y análisis EDA.|

## Conclusión: ¿Cuándo usar cada uno?

- **Matplotlib:** Para gráficos altamente personalizados y completos.
    
- **Seaborn:** Para análisis rápido de datos con gráficos 