TODO
Comentar y revista todo el codigo para personalizarlo

Paso a paso procedimientos:

- datos tomados del dataset de kaggle de pacientes de cancer de colon (https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions)

- analizar dataset para su uso en el modelo, generar datos nuevos, eliminar datos innecesarios y limpiar datos.

- En el modelo analizar paso a paso que hace el modelo, parametros de entrada, resultados dados validaciones, analisis de los datos con los que se entrena el modelo para ver si los resultados son los esperados.


dataset improvements:
- age -> generar registros de gente joven 30-50 años, generar muy pocos un 5-10% del total. Los demas generar casos de gente sin cancer de colon en el rango de edad donde ya lo podria concedir.
- Economic_Classification / Healthcare_Access / Insurance_Status -> se usan para calculos de valores utiles pero eliminar obiar en el modelo, son innecesarias para la prediccion de cancer.
