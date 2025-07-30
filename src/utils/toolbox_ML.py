import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, ttest_ind
from scipy.stats import pearsonr

def describe_df(df):
    """
    Recibe un dataframe y devuelve otro dataframe con los nombres 
    de las columnas del dataframe transferido a la función. En filas contiene 
    los parámetros descriptivos del dataframe: 
    - Tipo de dato
    - % Nulos (Porcentaje de valores nulos)
    - Valores Únicos
    - % Cardinalidad (Relación de valores únicos con el total de registros)
    
    Argumentos:
    - df (DataFrame): DataFrame de trabajo
    
    Retorna:
    - DataFrame con los parámetros descriptivos
    """
    
    summary_df = pd.DataFrame(index=['Tipo', '% Nulos', "Valores infinitos", 'Valores Únicos', '% Cardinalidad'])

    for column in df.columns:
        tipo = df[column].dtype
        porcentaje_nulos = df[column].isnull().mean() * 100
        verificar_si_es_numerico = np.issubdtype(df[column].dtype,np.number)
        if (verificar_si_es_numerico):
            valores_inf = ("Yes" if np.isinf(df[column]).any() else "No")
        else:
            valores_inf = "No"
        valores_unicos = df[column].nunique()
        cardinalidad = (valores_unicos / len(df)) * 100

        summary_df[column] = [tipo, f"{porcentaje_nulos:.2f}%",valores_inf, valores_unicos, f"{cardinalidad:.2f}%"]

    return summary_df


def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.2):
    """
    Clasifica las columnas de un DataFrame según su tipo de variable: Binaria, Categórica, Numérica Discreta o Continua.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    umbral_categoria (int): Máximo número de valores únicos para que una variable sea considerada categórica.
    umbral_continua (float): Porcentaje mínimo (sobre total de filas) para considerar una variable como continua.

    Devuelve:
    pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
    """
    resultado = []
    n = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = round(cardinalidad / n, 2)

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"

        resultado.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultado)

def get_features_num_regression(df,target_col,umbral_corr,pvalue=None,mostrar=False):
    """
    Conseguir la lista de features que tienen gran impacto en la target.
    Argumentos:
        df:DataFrame que se pasa de entrada
        target_col:la variable target con el que se quiere analizar,tiene que ser numerico
        umbral_corr: Umbral minimo para considerar una variable importante para el modelo
        pvalue: Certeza estadística con la que queremos validar la importancia de las feature
    Returns:
        Lista:Lista de features importantes.
        Mostrar: Muestra la matriz de correlación en una grafica HeatMap.
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna target {target_col} no esta en el DataFrame dado.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El dato de entrada tiene que ser un DataFrame.")
    
    if umbral_corr < 0 or umbral_corr > 1:
        raise ValueError("Umbral de correlacion tiene que estar entre 0 y 1")
    if pvalue is not None and (pvalue < 0 or pvalue > 1):
        raise ValueError("P-value tiene que estar entre 0 y 1")

    if not (np.issubdtype(df[target_col].dtype,np.number)):
        raise TypeError(f"Columna target {target_col} tiene que ser numerico amigo")
    
    cardinalidad_target = df[target_col].nunique() / len(df) * 100
    if cardinalidad_target < 10:
        warnings.warn(f"Columna target {target_col} tiene poca cardinalidad ({cardinalidad_target:.2f}%).")
    if cardinalidad_target > 95:
        warnings.warn(f"Columna target {target_col} tiene mucha cardinalidad ({cardinalidad_target:.2f}%).")
    if cardinalidad_target == 100:
        raise ValueError(f"Columna target {target_col} tiene 100% cardinalidad.")

    if df[target_col].isnull().sum() > 0:
        raise ValueError(f"Columna target {target_col} tiene valores Nulos.")
    
    corr = df.corr(numeric_only=True)[target_col]
    if mostrar:
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation heatmap for {target_col}")
        plt.show()
    
    corr = corr[abs(corr) > umbral_corr]
    corr = corr.drop(target_col)
    lista = []
    if pvalue is not None:
        pvalues = []
        for col in corr.index:
            _, p = pearsonr(df[target_col], df[col])
            if p < pvalue:
                lista.append(col)
    return lista


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera pairplots con variables numéricas del dataframe que cumplan ciertas condiciones de correlación con target_col.

    Parámetros:
    - df (DataFrame): DataFrame de entrada.
    - target_col (str): Columna objetivo para el análisis de correlación.
    - columns (list): Lista de columnas a considerar; si está vacía, se tomarán todas las numéricas del dataframe.
    - umbral_corr (float): Umbral mínimo absoluto de correlación para incluir variables.
    - pvalue (float or None): Nivel de significación estadística para el test de correlación. Si es None, no se aplica.

    Retorna:
    - Lista de columnas seleccionadas según las condiciones.
    """

    # Validación de target_col (debe existir enel dataframe)
    if target_col not in df.columns:
        raise ValueError(f"La columna target_col '{target_col}' no existe en el dataframe.")

    # Si columns está vacío, tomamos todas las variables numéricas excepto target_col
    if not columns:
        columns = df.select_dtypes(include=np.number).columns.tolist() #añado todas las columnas numéricas a la lista de columnas vacía
        columns.remove(target_col) #le quito el target
    #si hay columnas en parámetros tomará esas

    # Filtrar columnas por correlación
    selected_columns = [] #creo una lista vacía de columnas seleccionadas que iré rellenando
    for col in columns: #recorro las columnas de columns
        if col == target_col: #si la col es el target me la salto
            continue
        corr = df[[target_col, col]].dropna().corr().iloc[0, 1]  # tomo las columnas target_col y col del dataframe, elimina los NaN y calcula la matriz de correlación y extrae el valor con el iloc
        
        if abs(corr) > umbral_corr: #si la correlación en valor absoluto es mayor del umbral
            # Si se especifica pvalue, verificar significación estadística
            if pvalue is not None: #si el pvalue no es None
                _, pval = pearsonr(df[target_col].dropna(), df[col].dropna()) #calculo la correlación entre target_col y col y devuelve el vp_val porque el corr_coef no me hace falta
                if pval < 1 - pvalue: #si la probabilidad pval de que la correlación ocurra al azar es menor de 1-pvalue
                    selected_columns.append(col) #es estadísticamente signifcativa y lo meto en la lista
            else:
                selected_columns.append(col) # si no hay pvalue agrega la columna a la lista para verificarla

    # Graficar en grupos de máximo 5 columnas por gráfico
    if selected_columns: #si selected_columns no está vacía
        for i in range(0, len(selected_columns), 4):  # Genero números e 0 a la longitud de selected_columns de 4 en 4. Máximo 5 con target_col, proceso 4 columnas de cada iteración
            subset = [target_col] + selected_columns[i:i+4] #creo este subset que tiene el target y las 4 columnas
            sns.pairplot(df[subset].dropna(), diag_kind='kde') #hago el pairplot habiendo eliminado las filas con Nan con el dropna
            plt.show() #lo muestro

    return selected_columns #devuelvo las columnas que superaron el filtro de correlación y significancia

# Ejemplo de uso:
data = {
    'target': [1, 2, 3, 4, 5, 6, 7],
    'A': [2, 4, 6, 8, 10, 12, 14],
    'B': [1, 3, 3, 5, 5, 7, 7],
    'C': [5, 4, 3, 2, 1, 0, -1],
    'D': [10, 20, 30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)

result = plot_features_num_regression(df, target_col="target", umbral_corr=0.5, pvalue=0.05)
print("Columnas seleccionadas:", result)

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve una lista de columnas categóricas que presentan una relación significativa
    con la variable numérica target_col usando t-test o ANOVA según corresponda.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (numérica continua o discreta con alta cardinalidad).
    pvalue (float): Nivel de significación estadística (default = 0.05).

    Retorna:
    list or None: Lista de variables categóricas relacionadas, o None si hay error en los argumentos.
    """
    
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("❌ 'df' debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"❌ La columna '{target_col}' no está en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"❌ La columna '{target_col}' no es numérica.")
        return None

    if not (0 < pvalue < 1):
        print("❌ 'pvalue' debe estar entre 0 y 1.")
        return None

    cardinalidad = df[target_col].nunique()
    porcentaje = cardinalidad / len(df)

    if cardinalidad < 10 or porcentaje < 0.05:
        print(f"❌ La variable '{target_col}' no tiene suficiente cardinalidad para considerarse continua.")
        print(f"Cardinalidad única: {cardinalidad} ({round(porcentaje * 100, 2)}%)")
        return None

    # Selección de columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        print("⚠️ No hay variables categóricas en el DataFrame.")
        return []

    relacionadas = []

    for col in cat_cols:
        niveles = df[col].dropna().unique()
        grupos = [df[df[col] == nivel][target_col].dropna() for nivel in niveles]

        if any(len(grupo) < 2 for grupo in grupos):
            continue  # no hay suficientes datos en alguno de los grupos

        try:
            if len(niveles) == 2:
                stat, p = ttest_ind(*grupos)
            elif len(niveles) > 2:
                stat, p = f_oneway(*grupos)
            else:
                continue

            if p < pvalue:
                relacionadas.append(col)
        except Exception as e:
            print(f"⚠️ Error evaluando la columna '{col}': {e}")
            continue

    return relacionadas

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):

    """
    Identifica variables categóricas que tienen una relación significativa con una variable
    numérica continua usando ANOVA de una vía. Opcionalmente, genera histogramas agrupados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.

    target_col : str
        Nombre de la columna numérica continua a predecir.

    columns : list of str, opcional
        Columnas categóricas a evaluar. Si está vacío, se detectan automáticamente.

    pvalue : float, opcional
        Umbral de significancia (por defecto 0.05).

    with_individual_plot : bool, opcional
        Si es True, se grafican los histogramas por categoría.

    Retorna
    -------
    list of str
    Columnas categóricas significativamente relacionadas con la variable objetivo.

    """
    
    # Validación de DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None
    
    # Validación de target_col
    if not target_col or target_col not in df.columns:
        print("Error: target_col no está en el DataFrame o es vacío.")
        return None
    
    # Validación de tipo de target_col
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una variable numérica continua.")
        return None

    # Validación de columns
    if not isinstance(columns, list):
        print("Error: columns debe ser una lista de strings.")
        return None
    
    # Si columns está vacío, seleccionamos categóricas automáticamente
    if not columns:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    columnas_significativas = []

    for col in columns:
        if col not in df.columns:
            print(f"Aviso: La columna '{col}' no está en el DataFrame. Se omite.")
            continue

        if df[col].nunique() <= 1:
            continue  

        try:
            grupos = [df[df[col] == cat][target_col].dropna() for cat in df[col].dropna().unique()]
            if any(len(g) == 0 for g in grupos):
                continue

            f_stat, p_val = f_oneway(*grupos)

            if p_val < pvalue:
                columnas_significativas.append(col)

                if with_individual_plot:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data=df, x=target_col, hue=col, multiple="stack", kde=False)
                    plt.title(f"{col} vs {target_col} (p = {p_val:.4f})")
                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            print(f"Error evaluando la columna '{col}': {e}")

    return columnas_significativas

