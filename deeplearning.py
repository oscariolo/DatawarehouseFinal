import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CACHE_DIR = Path("data_cache")
CACHE_FILE = CACHE_DIR / "fact_emigrante_fact_inmigrante.parquet"
TARGET_COLUMN = "dim_ocupacion_ocu_migr"
N_CLASSES = 0 # Will be determined from data
INPUT_SHAPE = 0 # Will be determined from data

# --- 1. Data Loading ---
def load_data_from_cache(cache_path: Path = CACHE_FILE):
    """
    Loads the dataset from a Parquet cache file.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}. Please run machineLearning.py to generate it.")
    print(f"📦 Loading cached dataset from: {cache_path}")
    df = pd.read_parquet(cache_path)
    # Quick fix for column name if needed, as seen in machineLearning.py
    if 'dim_ocupacion_ocu_migr' not in df.columns and 'ocu_migr' in df.columns:
        df = df.rename(columns={'ocu_migr': 'dim_ocupacion_ocu_migr'})
    return df

def preprocess_for_pytorch(df: pd.DataFrame, target_column: str):
    """
    Preprocesses the data for a PyTorch model:
    - Handles categorical and numerical features.
    - Encodes the target variable.
    - Splits data into training and testing sets.
    - Returns tensors for train and test sets.
    """
    print("⚙️ Preprocessing data for PyTorch...")

    # Drop identifiers and irrelevant columns
    # Taking cues from machineLearning.py
    df_clean = df.drop(columns=[col for col in df.columns if 'id' in col.lower() and col != target_column] + ['source_fact', 'dim_fecha_fecha_completa'], errors='ignore')

    # Handle missing values for the target column
    df_clean = df_clean.dropna(subset=[target_column])
    
    # Separate features and target
    X = df_clean.drop(columns=[target_column])
    y_raw = df_clean[target_column]

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    global N_CLASSES
    N_CLASSES = len(label_encoder.classes_)
    print(f"Target variable '{target_column}' encoded into {N_CLASSES} classes.")

    # Identify feature types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Preprocess features: One-hot encode categorical and scale numerical
    X_processed = pd.get_dummies(X, columns=categorical_features, dummy_na=True)
    
    scaler = StandardScaler()
    X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
    
    # Fill any remaining NaN values (e.g., from dummy_na columns if they only appear in test)
    X_processed = X_processed.fillna(0)

    global INPUT_SHAPE
    INPUT_SHAPE = X_processed.shape[1]
    print(f"Input shape for the model: {INPUT_SHAPE} features.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder

# --- 2. Model Definition ---
class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn=nn.ReLU()):
        """
        Defines the architecture of the neural network.
        
        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): A list where each element is the number of neurons in a hidden layer.
            output_size (int): Number of output classes.
            activation_fn (nn.Module): The activation function to use between layers.
        """
        super(ClassificationNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Dynamically create hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(0.3)) # Add dropout for regularization
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def build_model(
    input_size=INPUT_SHAPE,
    layers=[128, 64],
    neurons=[128, 64], # Kept for compatibility if user thinks in neurons per layer
    activation_function=nn.ReLU(),
    output_size=N_CLASSES
    ):
    """
    Helper function to construct the model with specific parameters.
    """
    # Let `layers` take precedence if provided
    hidden_layers = layers if layers else neurons
    print(f"🏗️ Building model with architecture: Input({input_size}) -> {hidden_layers} -> Output({output_size})")
    
    return ClassificationNet(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        activation_fn=activation_function
    )

# --- 3. Training Loop ---
def train_model(
    model,
    X_train, y_train,
    epochs=25,
    learning_rate=0.001,
    batch_size=64,
    backpropagation_strategy=optim.Adam
    ):
    """
    Trains the PyTorch model.
    """
    print("🚀 Starting model training...")
    model.to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = backpropagation_strategy(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("✅ Training finished.")
    return model

# --- 4. Performance Analysis ---
@torch.no_grad() # Disable gradient calculation for evaluation
def analyze_performance(model, X_test, y_test, label_encoder):
    """
    Analyzes the model's performance using various metrics and plots the ROC curve.
    """
    print("\n📈 Analyzing model performance...")
    model.to(device)
    X_test = X_test.to(device)
    
    model.eval() # Set model to evaluation mode
    
    # Get predictions
    outputs = model(X_test)
    _, y_pred = torch.max(outputs, 1)
    y_probs = torch.softmax(outputs, dim=1)

    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()
    
    # --- Classification Metrics ---
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    # --- ROC Curve for Multi-Class ---
    # Binarize the output labels for ROC curve calculation
    y_test_bin = label_binarize(y_test_np, classes=range(N_CLASSES))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
             
    # Plot macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= N_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)

    colors = plt.cm.get_cmap('hsv', N_CLASSES)
    for i, color in zip(range(N_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_plot_path = 'roc_curve_multiclass.png'
    plt.savefig(roc_plot_path)
    print(f"Saved multi-class ROC curve plot to '{roc_plot_path}'.")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and preprocess data
    try:
        raw_df = load_data_from_cache()
        X_train, y_train, X_test, y_test, encoder = preprocess_for_pytorch(raw_df, TARGET_COLUMN)
        
        # Move tensors to the selected device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        # Update globals now that data is processed
        # This is not ideal, but necessary given the script structure
        # A class-based approach would be cleaner.
        if 'INPUT_SHAPE' not in globals() or INPUT_SHAPE == 0:
             INPUT_SHAPE = X_train.shape[1]
        if 'N_CLASSES' not in globals() or N_CLASSES == 0:
             N_CLASSES = len(encoder.classes_)


        # 2. Build the model with a custom architecture
        # You can easily edit layers, neurons, and activation functions here
        model = build_model(
            input_size=INPUT_SHAPE,
            layers=[256, 128, 64],  # Customize the layer architecture
            activation_function=nn.Tanh(), # e.g., nn.ReLU(), nn.LeakyReLU()
            output_size=N_CLASSES
        )
        print(model)

        # 3. Train the model
        # You can choose a different optimizer if needed
        trained_model = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            epochs=20, # Adjust number of epochs
            learning_rate=0.005,
            backpropagation_strategy=optim.AdamW # e.g., optim.SGD, optim.RMSprop
        )

        # 4. Analyze the performance on the test set
        analyze_performance(trained_model, X_test, y_test, encoder)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
