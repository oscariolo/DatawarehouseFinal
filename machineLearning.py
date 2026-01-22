import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy.exc import OperationalError
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, adjusted_rand_score, silhouette_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- Database Connection Constants ---
# PLEASE REPLACE WITH YOUR ACTUAL DATABASE DETAILS
DB_USER = "pentaho"
DB_PASSWORD = "password123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "pentaho_db"

# Construct the database URI
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# TABLES SCHEMA

FACT_TABLES = {
    "fact_inmigrante": {
        "columns": [
            "id_persona",
            "id_transporte",
            "id_frontera",
            "id_ocupacion",
            "id_fecha"
        ]
    },
    "fact_emigrante": {
        "columns": [
            "id_persona",
            "id_transporte",
            "id_frontera",
            "id_ocupacion",
            "id_fecha"
        ]
    }
}

DIM_TABLES = {
    "dim_persona": {
        "key": "id_persona",
        "columns": ["sex_migr", "nac_migr"]
    },
    "dim_transporte": {
        "key": "id_transporte",
        "columns": ["via_tran"]
    },
    "dim_frontera": {
        "key": "id_frontera",
        "columns": ["jef_migr", "pro_jefm","can_jefm"]
    },
    "dim_ocupacion": {
        "key": "id_ocupacion",
        "columns": ["ocu_migr"]
    },
    "dim_fecha": {
        "key": "id_fecha",
        "columns": ["fecha_completa", "anio_movi", "mes_movi","dia_movi"]
    }
}

from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y):
    """
    Computes balanced class weights as a dictionary {class_label: weight}.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weight_dict = dict(zip(classes, weights))

    print("\nClass distribution:")
    print(pd.Series(y).value_counts())

    print("\nComputed class weights:")
    print(class_weight_dict)

    return class_weight_dict


def build_fact_query(fact_table: str, limit=None) -> str:
    fact_cols = FACT_TABLES[fact_table]["columns"]
    
    select_cols = [f"{fact_table}.{c}" for c in fact_cols]

    joins = []
    for dim, cfg in DIM_TABLES.items():
        key = cfg["key"]
        for col in cfg["columns"]:
            select_cols.append(f"{dim}.{col} AS {dim}_{col}")
        joins.append(
            f"LEFT JOIN {dim} ON {fact_table}.{key} = {dim}.{key}"
        )

    sql = f"""
    SELECT
        {', '.join(select_cols)}
    FROM {fact_table}
    {' '.join(joins)}
    """

    if limit:
        sql += f" LIMIT {limit}"

    return sql

from pathlib import Path

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_or_build_dataset(
    db_uri: str,
    fact_tables: list,
    limit=None,
    force_reload=False
) -> pd.DataFrame:
    cache_key = "_".join(sorted(fact_tables))
    cache_file = CACHE_DIR / f"{cache_key}.parquet"

    if cache_file.exists() and not force_reload:
        print(f"📦 Loading cached dataset: {cache_file}")
        df = pd.read_parquet(cache_file)
        #limitar a definido
        if(limit is not None):
            print("Sampling dataset to " + str(limit) + " rows...")
            df = df.sample(n=limit, random_state=42)
            return df
        else:
            return df

    print("🔌 Connecting to database...")
    engine = create_engine(db_uri)

    dfs = []
    with engine.connect() as conn:
        for fact in fact_tables:
            print(f"⚙ Building dataset for {fact}")
            sql = build_fact_query(fact, limit)
            df = pd.read_sql_query(sql, conn)
            df["source_fact"] = fact
            dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    print(f"💾 Saving processed dataset → {cache_file}")
    final_df.to_parquet(
        cache_file,
        engine="pyarrow",
        compression="snappy"
    )

    return final_df


# --- Configuration Constants ---
TABLE_NAME = ["fact_inmigrante", "fact_emigrante"]  # Replace with your table name or list of tables
TASK = "classification"             # "clustering", "classification", or "regression"
TARGET_COLUMN = "dim_ocupacion_ocu_migr"            # Required for classification, e.g., 'some_column'
LIMIT = 1000000             # Set to an integer to limit rows, or None
EXCLUDE_YEARS = [2020]
# -----------------------------------

def get_data(db_uri, table_names, limit=None):
    """
    Connects to the PostgreSQL database and fetches data.
    If the table is a known fact table (e.g., 'fact_inmigrante', 'fact_emigrante'), 
    it performs joins with dimension tables.
    Supports loading multiple tables if a list is provided.

    Args:
        db_uri (str): The database connection URI.
        table_names (str or list): The name(s) of the table(s) to fetch.
        limit (int, optional): The maximum number of rows to fetch.

    Returns:
        pandas.DataFrame: The data from the table(s).
    """
    if isinstance(table_names, str):
        table_list = [table_names]
    else:
        table_list = table_names

    all_dfs = []

    try:
        engine = create_engine(db_uri)
        print("Connecting to the database...")
        with engine.connect() as connection:
            for table_name in table_list:
                is_fact_table = table_name in ['fact_inmigrante', 'fact_emigrante']
                
                if is_fact_table:
                    print(f"Fact table '{table_name}' detected. Building join query...")
                    # Based on the provided schema, the foreign keys are:
                    # id_persona, id_transporte, id_frontera, id_ocupacion, id_fecha
                    dims = {
                        'dim_persona': 'id_persona',
                        'dim_transporte': 'id_transporte',
                        'dim_frontera': 'id_frontera',
                        'dim_ocupacion': 'id_ocupacion',
                        'dim_fecha': 'id_fecha',
                    }
                    
                    # This query joins the fact table with all dimension tables.
                    # We use SELECT * and then remove duplicated columns in pandas.
                    sql_query = f"SELECT * FROM {table_name} "
                    for dim_table, key in dims.items():
                        sql_query += f"LEFT JOIN {dim_table} ON {table_name}.{key} = {dim_table}.{key} "

                    if limit:
                        sql_query += f" LIMIT {limit}"

                    print(f"Executing join query for '{table_name}'...")
                    df = pd.read_sql_query(sql_query, connection)
                    
                    # Remove duplicated columns that result from the joins (e.g., id columns).
                    df = df.loc[:,~df.columns.duplicated()]

                else:
                    print(f"Reading data from single table '{table_name}'...")
                    if limit:
                        sql_query = f"SELECT * FROM {table_name} LIMIT {limit}"
                        df = pd.read_sql_query(sql_query, connection)
                    else:
                        df = pd.read_sql_table(table_name, connection)
                
                all_dfs.append(df)
            
            if not all_dfs:
                return None
                
            final_df = pd.concat(all_dfs, ignore_index=True)
            print("Data loaded successfully.")
            print(f"Combined DataFrame contains {len(final_df.columns)} columns and {len(final_df)} rows.")
            final_df.to_pickle('combined_data.pkl')
            return final_df
    except OperationalError as e:
        print(f"\n[!] Connection Failed: {e}")
        if "password authentication failed" in str(e):
            print("[!] Hint: You are loading a dump file. Did the dump file overwrite the password?")
            print("[!] Try using the password from the original database, or remove 'CREATE/ALTER ROLE' from the .sql file.")
        return None
    except Exception as e:
        print(f"An error occurred while connecting to the database or reading data: {e}")
        return None


def preprocess_data(df):
    # Explicitly drop only true identifiers
    id_cols = [
        c for c in df.columns
        if c.startswith("id_")
    ]
    df = df.drop(columns=id_cols, errors="ignore")
    df = df.drop(columns=["dim_fecha_fecha_completa"], errors="ignore")

    print(f"Dropped ID columns: {id_cols}")

    # Identify feature types
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric features to be scaled: {numeric_features}")
    print(f"Categorical features to be one-hot encoded: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"   
    )

    return preprocessor

def perform_clustering(df, applyAnalysis=False):
    """
    Performs KMeans clustering on the data.
    """
    print("\n--- Performing Clustering ---")    
    if df is None or df.empty:
        print("DataFrame is empty. Cannot perform clustering.")
        return

    preprocessor = preprocess_data(df)
    
    # Transform data once for the search
    print("Preprocessing data for clustering search...")
    X_processed = preprocessor.fit_transform(df)

    # --- Elbow Method & Silhouette Analysis ---
    if(applyAnalysis):
        print("Running Elbow Method and Silhouette Analysis to find optimal k...")
        inertias = []
        silhouette_scores = []
        K_range = range(2, 11) # Test k from 2 to 10

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_processed)
            inertias.append(kmeans.inertia_)
            
            # Silhouette score requires at least 2 clusters and size > k
            if X_processed.shape[0] > k:
                score = silhouette_score(X_processed, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)

        # Plotting metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        plt.subplot(1, 2, 2)
        plt.plot(K_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores')
        plt.tight_layout()
        plt.savefig('clustering_metrics.png')
        plt.close()
        print("Saved clustering metrics plot to 'clustering_metrics.png'.")
        # Select best k (simple heuristic: max silhouette score)
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters determined: {best_k}")

    else:
        best_k = 3

    
    # --- Final Model Training ---
    print(f"Fitting final KMeans model with k={best_k}...")
    final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    final_kmeans.fit(X_processed)
    labels = final_kmeans.labels_
    
    # ---Output the samples for analysis----
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    print("\n--- Sample of 10 rows per cluster ---")
    sample_per_cluster = (
        df_clustered
        .groupby('cluster', group_keys=False)
        .apply(lambda x: x.sample(n=min(10, len(x)), random_state=42))
    )
    
    sample_per_cluster.to_csv("cluster_samples.csv", index=False)
    print("Saved cluster samples to 'cluster_samples.csv'.")

    # --- Visualization ---
    print("Generating clustering visualization...")
    
    # Convert to dense if sparse (PCA requires dense input)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    # Create a DataFrame for plotting
    df_viz = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_viz['Cluster'] = labels

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=50)
    plt.title('Clustering Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    
    # Save the plot
    plot_filename = 'clustering_blobs.png'
    plt.savefig(plot_filename)
    print(f"Clustering graph saved as '{plot_filename}'.")
    plt.close()

def prepare_classification_data(df, target_column):
    """
    Prepares the data for classification: splits into X/y, train/test, and creates preprocessor.
    """
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None, None, None

    # Drop ID columns before splitting, target might be an ID column
    id_cols = [col for col in df.columns if 'id' in col.lower() and col != target_column]
    id_cols.append("source_fact")
    id_cols.append("dim_fecha_fecha_completa")
    
    df_clean = df.drop(columns=id_cols, errors='ignore')

    X = df_clean.drop(target_column, axis=1)
    y = df_clean[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numeric features for classification: {list(numeric_features)}")
    print(f"Categorical features for classification: {list(categorical_features)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_specific_model(X_train, X_test, y_train, y_test, preprocessor, model_name='RandomForest', params=None):
    """
    Trains a specific model specified by model_name without grid search.
    """
    print(f"\n--- Training Specific Model: {model_name} ---")
    
    class_weight = compute_class_weights(y_train)
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weight),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight),
        'SVM': SVC(random_state=42,class_weight=class_weight),
        'DeepLearning_MLP': MLPClassifier(random_state=42, max_iter=500)
    }
    
    if model_name not in models:
        print(f"Error: Model '{model_name}' not recognized. Available models: {list(models.keys())}")
        return None

    model = models[model_name]
    if params:
        print(f"Setting parameters: {params}")
        model.set_params(**params)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    print("Model trained successfully.")

    print("\nEvaluation on Test Set:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return pipeline

def perform_grid_search(X_train, X_test, y_train, y_test, preprocessor):
    """
    Performs Grid Search across multiple models to find the best one.
    """
    print("\n--- Starting Grid Search for model selection ---")
    
    # Define models and their hyperparameter grids
    models_config = [
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20]
            }
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10]
            }
        },
        {
            'name': 'SVM',
            'model': SVC(random_state=42),
            'params': {
                'classifier__C': [0.1, 1],
                'classifier__kernel': ['linear', 'rbf']
            }
        },
        {
            'name': 'DeepLearning_MLP',
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': [0.0001, 0.05]
            }
        }
    ]

    best_global_score = -1
    best_global_model = None
    best_global_name = ""

    for config in models_config:
        print(f"\nTesting model: {config['name']}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', config['model'])])
        
        # Use GridSearchCV
        grid = GridSearchCV(pipeline, config['params'], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Best CV Accuracy: {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_global_score:
            best_global_score = grid.best_score_
            best_global_model = grid.best_estimator_
            best_global_name = config['name']

    print(f"\n--- Best Model Selected: {best_global_name} ---")
    print(f"Best Validation Accuracy: {best_global_score:.4f}")

    print("\nFinal Evaluation on Test Set:")
    y_pred = best_global_model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return best_global_model

def perform_classification(df, target_column):
    """
    Wrapper function to prepare data and run classification.
    """
    print("\n--- Performing Classification ---")
    if df is None or df.empty:
        print("DataFrame is empty. Cannot perform classification.")
        return
        
    X_train, X_test, y_train, y_test, preprocessor = prepare_classification_data(df, target_column)
    
    if X_train is None:
        return

    # To train a specific model directly, uncomment the line below:
    train_specific_model(X_train, X_test, y_train, y_test, preprocessor, model_name='DeepLearning_MLP')
    
    # Default: Perform Grid Search
    #perform_grid_search(X_train, X_test, y_train, y_test, preprocessor)

def perform_regression_analysis(df, exclude_years=[2020, 2021]):
    """
    Performs regression analysis to predict movement based on dates.
    Aggregates data by date and forecasts future movement.
    """
    print("\n--- Performing Regression Analysis (Forecasting) ---")
    
    # Ensure date columns exist (based on star schema description)
    # We look for 'fecha_completa' and date parts.
    if 'fecha_completa' not in df.columns:
        print("Error: 'fecha_completa' column not found for regression.")
        return

    # Convert to datetime
    df['fecha_completa'] = pd.to_datetime(df['fecha_completa'])
    
    # Filter out outlier year if specified
    if exclude_years and 'anio_movi' in df.columns:
        print(f"Excluding data for years: {exclude_years}")
        for year in exclude_years:
            df = df[df['anio_movi'] != year]
    
    # Aggregate data to get daily movement counts
    print("Aggregating data by date...")
    daily_counts = df.groupby('fecha_completa').size().reset_index(name='movement_count')
    daily_counts = daily_counts.sort_values('fecha_completa')
    
    # Feature Engineering
    daily_counts['ordinal_date'] = daily_counts['fecha_completa'].apply(lambda x: x.toordinal())
    daily_counts['month'] = daily_counts['fecha_completa'].dt.month
    daily_counts['day'] = daily_counts['fecha_completa'].dt.day
    
    X = daily_counts[['ordinal_date', 'month', 'day']]
    y = daily_counts['movement_count']
    
    # Split data (Time-based split)
    split_idx = int(len(daily_counts) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train = daily_counts['fecha_completa'].iloc[:split_idx]
    dates_test = daily_counts['fecha_completa'].iloc[split_idx:]
    
    # Train Model (Linear Regression for trend extrapolation)
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:\n  Mean Squared Error: {mse:.2f}\n  R2 Score: {r2:.4f}")
    
    # Predict for following years (Next 365 days)
    print("Forecasting for the next year...")
    last_date = daily_counts['fecha_completa'].max()
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, 366)]
    future_df = pd.DataFrame({'fecha_completa': future_dates})
    future_df['ordinal_date'] = future_df['fecha_completa'].apply(lambda x: x.toordinal())
    future_df['month'] = future_df['fecha_completa'].dt.month
    future_df['day'] = future_df['fecha_completa'].dt.day
    
    y_future_pred = model.predict(future_df[['ordinal_date', 'month', 'day']])
    
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.scatter(dates_train, y_train, color='blue', label='Training Data', alpha=0.3, s=10)
    plt.scatter(dates_test, y_test, color='green', label='Test Data', alpha=0.3, s=10)
    plt.plot(dates_test, y_pred, color='red', linewidth=2, label='Predictions (Test)')
    plt.plot(future_df['fecha_completa'], y_future_pred, color='orange', linestyle='--', linewidth=2, label='Future Forecast')
    
    plt.title(f'Movement Prediction (Regression) - Excluding {exclude_years}')
    plt.xlabel('Date')
    plt.ylabel('Movement Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_forecast.png')
    print("Saved regression plot to 'regression_forecast.png'.")

def main():
    df = load_or_build_dataset(
        db_uri=DB_URI,
        fact_tables=TABLE_NAME,  # ["fact_inmigrante", "fact_emigrante"]
        limit=LIMIT,
        force_reload=False  # set True if schema changes
    )
    
    print(df.shape)
    print(df.head())

    if TASK == "clustering":
        perform_clustering(df)
    elif TASK == "classification":
        perform_classification(df, TARGET_COLUMN)
    elif TASK == "regression":
        perform_regression_analysis(df, exclude_years=EXCLUDE_YEARS)


if __name__ == "__main__":
    main()
    print("\nScript finished.")