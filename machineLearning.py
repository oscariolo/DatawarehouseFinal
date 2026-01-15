import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy.exc import OperationalError
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, adjusted_rand_score, silhouette_score
from sklearn.impute import KNNImputer, SimpleImputer
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

# --- Configuration Constants ---
TABLE_NAME = "fact_inmigrante"  # Replace with your table name
TASK = "classification"             # "clustering" or "classification"
TARGET_COLUMN = "ocu_migr"            # Required for classification, e.g., 'some_column'
LIMIT = None                  # Set to an integer to limit rows, or None
# -----------------------------------

def get_data(db_uri, table_name, limit=None):
    """
    Connects to the PostgreSQL database and fetches data.
    If the table is a known fact table (e.g., 'fact_inmigrante'), it performs joins with dimension tables.
    Otherwise, it fetches data from the specified single table.

    Args:
        db_uri (str): The database connection URI.
        table_name (str): The name of the fact table or a single table to fetch.
        limit (int, optional): The maximum number of rows to fetch.

    Returns:
        pandas.DataFrame: The data from the table(s).
    """
    is_fact_table = table_name in ['fact_inmigrante', 'fact_emigrante']

    try:
        engine = create_engine(db_uri)
        print("Connecting to the database...")
        with engine.connect() as connection:
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
            
            print("Data loaded successfully.")
            print(f"DataFrame contains {len(df.columns)} columns.")
            return df
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
    """
    Identifies numeric and categorical features and creates a preprocessor.
    This function will handle mixed types (strings and numbers) by applying
    StandardScaler to numeric columns and OneHotEncoder to categorical columns.
    """
    # Drop ID columns as they are just identifiers and not useful features
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols, errors='ignore')
    print(f"Dropped ID columns: {id_cols}")

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    print(f"Numeric features to be scaled: {list(numeric_features)}")
    print(f"Categorical features to be one-hot encoded: {list(categorical_features)}")

    # Define transformers with imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough') # Keep other columns if any

    return preprocessor

def perform_clustering(df):
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

    # --- Final Model Training ---
    print(f"Fitting final KMeans model with k={best_k}...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_kmeans.fit(X_processed)
    labels = final_kmeans.labels_
    
    print("\nClustering results (first 20 labels):")
    print(labels[:20])

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

def perform_classification(df, target_column):
    """
    Performs classification on the data using a RandomForestClassifier.
    """
    print("\n--- Performing Classification ---")
    if df is None or df.empty:
        print("DataFrame is empty. Cannot perform classification.")
        return
        
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Drop ID columns before splitting, target might be an ID column
    id_cols = [col for col in df.columns if 'id' in col.lower() and col != target_column]
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

    # --- Model Selection with Grid Search ---
    print("Starting Grid Search for model selection...")
    
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

def main():
    """
    Main function to run the machine learning tasks.
    """
    if TASK == "classification" and not TARGET_COLUMN:
        print("Error: TARGET_COLUMN is required for classification task.")
        return

    # Fetch and join data from the star schema
    df = get_data(DB_URI, TABLE_NAME, LIMIT)

    if df is not None and not df.empty:
        if TASK == "clustering":
            perform_clustering(df)
        elif TASK == "classification":
            perform_classification(df, TARGET_COLUMN)
    else:
        print("Could not load data. Aborting.")

if __name__ == "__main__":
    main()
    print("\nScript finished.")