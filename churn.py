import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the transaction data for churn prediction
    """
    print("Loading data...")
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    return data

def feature_engineering(data):
    """
    Create features for churn prediction, including Customer Lifetime Value (CLTV)
    """
    print("Starting feature engineering...")
    
    # Convert transaction_date to datetime
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['quantity', 'unit_price', 'discount_applied', 'grand_total', 
                       'net_price', 'customer_age', 'target_multiplier', 'latitude', 'longitude']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convert binary flags to numeric
    binary_columns = ['voucher_redeemed_flag', 'is_upsell', 'is_returned']
    for col in binary_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Create customer-level aggregated features
    agg_dict = {}
    
    # Numeric aggregations
    if 'quantity' in data.columns:
        agg_dict['quantity'] = ['sum', 'mean', 'count']
    if 'unit_price' in data.columns:
        agg_dict['unit_price'] = ['mean', 'max', 'min']
    if 'discount_applied' in data.columns:
        agg_dict['discount_applied'] = ['mean', 'sum']
    if 'grand_total' in data.columns:
        agg_dict['grand_total'] = ['sum', 'mean', 'max']
    if 'net_price' in data.columns:
        agg_dict['net_price'] = ['sum', 'mean']
    if 'is_returned' in data.columns:
        agg_dict['is_returned'] = ['sum', 'mean']
    if 'is_upsell' in data.columns:
        agg_dict['is_upsell'] = ['sum', 'mean']
    if 'voucher_redeemed_flag' in data.columns:
        agg_dict['voucher_redeemed_flag'] = ['sum', 'mean']
    
    # Date aggregations
    agg_dict['transaction_date'] = ['min', 'max', 'count']
    
    # Categorical aggregations (count unique)
    categorical_columns = ['product_id', 'product_category', 'branch_id', 'staff_id', 
                           'channel_id', 'payment_mode']
    for col in categorical_columns:
        if col in data.columns:
            agg_dict[col] = 'nunique'
    
    customer_features = data.groupby('customer_id').agg(agg_dict).reset_index()
    
    # Flatten column names
    new_columns = ['customer_id']
    for col in customer_features.columns[1:]:
        if isinstance(col, tuple):
            new_columns.append('_'.join(col).strip())
        else:
            new_columns.append(col)
    
    customer_features.columns = new_columns
    
    # Calculate derived features
    if 'transaction_date_min' in customer_features.columns and 'transaction_date_max' in customer_features.columns:
        customer_features['days_since_first_transaction'] = (
            pd.Timestamp.now() - customer_features['transaction_date_min']
        ).dt.days
        
        customer_features['days_since_last_transaction'] = (
            pd.Timestamp.now() - customer_features['transaction_date_max']
        ).dt.days
        
        # Calculate customer lifetime (days between first and last transaction)
        customer_features['customer_lifetime_days'] = (
            customer_features['transaction_date_max'] - customer_features['transaction_date_min']
        ).dt.days + 1  # Add 1 to avoid division by zero
    
    # Calculate transaction frequency
    if 'transaction_date_count' in customer_features.columns and 'customer_lifetime_days' in customer_features.columns:
        customer_features['transaction_frequency'] = (
            customer_features['transaction_date_count'] / customer_features['customer_lifetime_days']
        ).fillna(0)
    
    # Calculate Customer Lifetime Value (CLTV)
    # A simple CLTV model: (Total Revenue / Customer Lifetime in Days) * 365 (annualized)
    if 'grand_total_sum' in customer_features.columns and 'customer_lifetime_days' in customer_features.columns:
        # Avoid division by zero for customers with 0 lifetime days (e.g., single transaction)
        customer_features['customer_lifetime_value'] = np.where(
            customer_features['customer_lifetime_days'] > 0,
            (customer_features['grand_total_sum'] / customer_features['customer_lifetime_days']) * 365,
            customer_features['grand_total_sum'] # If lifetime is 0, CLTV is just total spent
        )
        customer_features['customer_lifetime_value'] = customer_features['customer_lifetime_value'].fillna(0)
    else:
        customer_features['customer_lifetime_value'] = 0 # Default if columns not found
    
    # Define churn target variable - MODIFIED CHURN CONDITIONS
    churn_conditions = []
    
    # Condition 1: No transaction in last 120 days (increased from 90)
    if 'days_since_last_transaction' in customer_features.columns:
        churn_conditions.append(customer_features['days_since_last_transaction'] > 120)  
    
    # Condition 2: Low transaction frequency (less than twice per 3 months ~ 0.02 transactions/day)
    if 'transaction_frequency' in customer_features.columns:
        churn_conditions.append(customer_features['transaction_frequency'] < 0.02)
    
    # Combine conditions (customer is churned if ANY condition is met)
    if churn_conditions:
        customer_features['churn'] = np.logical_or.reduce(churn_conditions).astype(int)
    else:
        # Fallback: use recency as main churn indicator
        customer_features['churn'] = (customer_features.get('days_since_last_transaction', 0) > 90).astype(int) # Fallback increased as well
    
    # Get customer age from original data (take first occurrence)
    if 'customer_age' in data.columns:
        age_data = data.groupby('customer_id')['customer_age'].first().reset_index()
        customer_features = customer_features.merge(age_data, on='customer_id', how='left')
    
    print(f"Feature engineering completed. Final shape: {customer_features.shape}")
    print(f"Churn distribution: {customer_features['churn'].value_counts().to_dict()}")
    
    return customer_features

def prepare_features_for_modeling(customer_features):
    """
    Prepare features for machine learning models
    """
    print("Preparing features for modeling...")
    
    # Dynamically select numeric feature columns (exclude customer_id, churn, and date columns)
    exclude_columns = ['customer_id', 'churn', 'transaction_date_min', 'transaction_date_max']
    feature_columns = [col for col in customer_features.columns 
                       if col not in exclude_columns and 
                       customer_features[col].dtype in ['int64', 'float64']]
    
    print(f"Selected features: {feature_columns}")
    
    # Create feature matrix
    X = customer_features[feature_columns].copy()
    y = customer_features['churn'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled, y, scaler, feature_columns

def train_models(X, y):
    """
    Train multiple ML models for churn prediction
    """
    print("Training machine learning models...")
    
    # Check if we have enough samples of each class
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    if len(class_counts) < 2:
        print("Warning: Only one class present in the data. Creating balanced dataset...")
        # If only one class, create some artificial minority class samples
        if class_counts.index[0] == 0:  # Only non-churned customers
            n_artificial = min(len(y) // 10, 50)  # Create 10% artificial churned customers, max 50
            artificial_indices = np.random.choice(y.index, n_artificial, replace=False)
            y.loc[artificial_indices] = 1
        else: # Only churned customers
            n_artificial = min(len(y) // 10, 50)
            artificial_indices = np.random.choice(y.index, n_artificial, replace=False)
            y.loc[artificial_indices] = 0
            
    # Split data with stratification if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails, do regular split (less ideal for imbalanced data)
        print("Warning: Stratification failed, performing non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            model_results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred, zero_division=0)
            }
            trained_models[name] = model
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    if not model_results:
        print("Error: No models could be trained successfully.")
        return None, None, None, None, None, None
    
    # Select best model (highest accuracy)
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest performing model: {best_model_name}")
    
    return best_model, X_train, X_test, y_train, y_test, best_model_name

def generate_predictions(model, customer_features, X_scaled, output_path):
    """
    Generate churn predictions and save to CSV
    """
    print("Generating predictions...")
    
    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of churn
    
    # Create results dataframe
    results = customer_features[['customer_id']].copy()
    results['actual_churn'] = customer_features['churn']
    results['predicted_churn'] = predictions
    results['churn_probability'] = prediction_probabilities
    # MODIFIED RISK LEVEL BINS
    results['risk_level'] = pd.cut(
        prediction_probabilities,
        bins=[0, 0.4, 0.75, 1.0], # Adjusted bins
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True # Include lowest value in first bin
    )
    
    # Add key customer metrics (if available)
    if 'transaction_date_count' in customer_features.columns:
        results['total_transactions'] = customer_features['transaction_date_count']
    if 'grand_total_sum' in customer_features.columns:
        results['total_spent'] = customer_features['grand_total_sum']
    if 'is_returned_mean' in customer_features.columns:
        results['return_rate'] = customer_features['is_returned_mean']
    if 'days_since_last_transaction' in customer_features.columns:
        results['days_since_last_transaction'] = customer_features['days_since_last_transaction']
    if 'customer_age' in customer_features.columns:
        results['customer_age'] = customer_features['customer_age']
    # Add Customer Lifetime Value
    if 'customer_lifetime_value' in customer_features.columns:
        results['customer_lifetime_value'] = customer_features['customer_lifetime_value']
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"Total customers: {len(results)}")
    print(f"Predicted churners: {results['predicted_churn'].sum()}")
    print(f"Churn rate: {results['predicted_churn'].mean():.2%}")
    print(f"\nRisk Level Distribution:")
    print(results['risk_level'].value_counts())
    
    return results

def main():
    """
    Main function to run the churn prediction pipeline
    """
    # File paths
    input_file = "D:\Mockup\synthetic_transaction_data.csv"
    output_file = "D:\Mockup\customer_churn_predictions.csv"
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data(input_file)
        
        # Feature engineering
        customer_features = feature_engineering(data)
        
        # Prepare features for modeling
        X_scaled, y, scaler, feature_columns = prepare_features_for_modeling(customer_features)
        
        # Train models
        result = train_models(X_scaled, y)
        if result[0] is None:
            print("Model training failed. Please check your data.")
            return
            
        best_model, X_train, X_test, y_train, y_test, best_model_name = result
        
        # Generate and save predictions
        results = generate_predictions(best_model, customer_features, X_scaled, output_file)
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features ({best_model_name}):")
            print(feature_importance.head(10))
        
        print("\n" + "="*50)
        print("CHURN PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_file}")
        print("Please ensure the file path is correct and the file exists.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your data format and file permissions.")

if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings('ignore')

# def load_and_preprocess_data(file_path):
#     """
#     Load and preprocess the transaction data for churn prediction
#     """
#     print("Loading data...")
#     data = pd.read_csv(file_path)
#     print(f"Data loaded successfully. Shape: {data.shape}")
    
#     return data

# def feature_engineering(data):
#     """
#     Create features for churn prediction
#     """
#     print("Starting feature engineering...")
    
#     # Convert transaction_date to datetime
#     data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    
#     # Ensure numeric columns are properly typed
#     numeric_columns = ['quantity', 'unit_price', 'discount_applied', 'grand_total', 
#                        'net_price', 'customer_age', 'target_multiplier', 'latitude', 'longitude']
    
#     for col in numeric_columns:
#         if col in data.columns:
#             data[col] = pd.to_numeric(data[col], errors='coerce')
    
#     # Convert binary flags to numeric
#     binary_columns = ['voucher_redeemed_flag', 'is_upsell', 'is_returned']
#     for col in binary_columns:
#         if col in data.columns:
#             data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
#     # Create customer-level aggregated features
#     agg_dict = {}
    
#     # Numeric aggregations
#     if 'quantity' in data.columns:
#         agg_dict['quantity'] = ['sum', 'mean', 'count']
#     if 'unit_price' in data.columns:
#         agg_dict['unit_price'] = ['mean', 'max', 'min']
#     if 'discount_applied' in data.columns:
#         agg_dict['discount_applied'] = ['mean', 'sum']
#     if 'grand_total' in data.columns:
#         agg_dict['grand_total'] = ['sum', 'mean', 'max']
#     if 'net_price' in data.columns:
#         agg_dict['net_price'] = ['sum', 'mean']
#     if 'is_returned' in data.columns:
#         agg_dict['is_returned'] = ['sum', 'mean']
#     if 'is_upsell' in data.columns:
#         agg_dict['is_upsell'] = ['sum', 'mean']
#     if 'voucher_redeemed_flag' in data.columns:
#         agg_dict['voucher_redeemed_flag'] = ['sum', 'mean']
    
#     # Date aggregations
#     agg_dict['transaction_date'] = ['min', 'max', 'count']
    
#     # Categorical aggregations (count unique)
#     categorical_columns = ['product_id', 'product_category', 'branch_id', 'staff_id', 
#                            'channel_id', 'payment_mode']
#     for col in categorical_columns:
#         if col in data.columns:
#             agg_dict[col] = 'nunique'
    
#     customer_features = data.groupby('customer_id').agg(agg_dict).reset_index()
    
#     # Flatten column names
#     new_columns = ['customer_id']
#     for col in customer_features.columns[1:]:
#         if isinstance(col, tuple):
#             new_columns.append('_'.join(col).strip())
#         else:
#             new_columns.append(col)
    
#     customer_features.columns = new_columns
    
#     # Calculate derived features
#     if 'transaction_date_min' in customer_features.columns and 'transaction_date_max' in customer_features.columns:
#         customer_features['days_since_first_transaction'] = (
#             pd.Timestamp.now() - customer_features['transaction_date_min']
#         ).dt.days
        
#         customer_features['days_since_last_transaction'] = (
#             pd.Timestamp.now() - customer_features['transaction_date_max']
#         ).dt.days
        
#         # Calculate customer lifetime (days between first and last transaction)
#         customer_features['customer_lifetime_days'] = (
#             customer_features['transaction_date_max'] - customer_features['transaction_date_min']
#         ).dt.days + 1  # Add 1 to avoid division by zero
    
#     # Calculate transaction frequency
#     if 'transaction_date_count' in customer_features.columns and 'customer_lifetime_days' in customer_features.columns:
#         customer_features['transaction_frequency'] = (
#             customer_features['transaction_date_count'] / customer_features['customer_lifetime_days']
#         ).fillna(0)
    
#     # Define churn target variable - MODIFIED CHURN CONDITIONS
#     churn_conditions = []
    
#     # Condition 1: No transaction in last 120 days (increased from 90)
#     if 'days_since_last_transaction' in customer_features.columns:
#         churn_conditions.append(customer_features['days_since_last_transaction'] > 120)   
    
#     # Condition 2: Low transaction frequency (less than twice per 3 months ~ 0.02 transactions/day)
#     if 'transaction_frequency' in customer_features.columns:
#         churn_conditions.append(customer_features['transaction_frequency'] < 0.02)
    
#     # Combine conditions (customer is churned if ANY condition is met)
#     if churn_conditions:
#         customer_features['churn'] = np.logical_or.reduce(churn_conditions).astype(int)
#     else:
#         # Fallback: use recency as main churn indicator
#         customer_features['churn'] = (customer_features.get('days_since_last_transaction', 0) > 90).astype(int) # Fallback increased as well
    
#     # Get customer age from original data (take first occurrence)
#     if 'customer_age' in data.columns:
#         age_data = data.groupby('customer_id')['customer_age'].first().reset_index()
#         customer_features = customer_features.merge(age_data, on='customer_id', how='left')
    
#     print(f"Feature engineering completed. Final shape: {customer_features.shape}")
#     print(f"Churn distribution: {customer_features['churn'].value_counts().to_dict()}")
    
#     return customer_features

# def prepare_features_for_modeling(customer_features):
#     """
#     Prepare features for machine learning models
#     """
#     print("Preparing features for modeling...")
    
#     # Dynamically select numeric feature columns (exclude customer_id, churn, and date columns)
#     exclude_columns = ['customer_id', 'churn', 'transaction_date_min', 'transaction_date_max']
#     feature_columns = [col for col in customer_features.columns 
#                        if col not in exclude_columns and 
#                        customer_features[col].dtype in ['int64', 'float64']]
    
#     print(f"Selected features: {feature_columns}")
    
#     # Create feature matrix
#     X = customer_features[feature_columns].copy()
#     y = customer_features['churn'].copy()
    
#     # Handle missing values
#     X = X.fillna(X.median())
    
#     # Remove any infinite values
#     X = X.replace([np.inf, -np.inf], np.nan)
#     X = X.fillna(X.median())
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = pd.DataFrame(
#         scaler.fit_transform(X),
#         columns=X.columns,
#         index=X.index
#     )
    
#     return X_scaled, y, scaler, feature_columns

# def train_models(X, y):
#     """
#     Train multiple ML models for churn prediction
#     """
#     print("Training machine learning models...")
    
#     # Check if we have enough samples of each class
#     class_counts = y.value_counts()
#     print(f"Class distribution: {class_counts.to_dict()}")
    
#     if len(class_counts) < 2:
#         print("Warning: Only one class present in the data. Creating balanced dataset...")
#         # If only one class, create some artificial minority class samples
#         if class_counts.index[0] == 0:  # Only non-churned customers
#             n_artificial = min(len(y) // 10, 50)  # Create 10% artificial churned customers, max 50
#             artificial_indices = np.random.choice(y.index, n_artificial, replace=False)
#             y.loc[artificial_indices] = 1
#         else: # Only churned customers
#             n_artificial = min(len(y) // 10, 50)
#             artificial_indices = np.random.choice(y.index, n_artificial, replace=False)
#             y.loc[artificial_indices] = 0
            
#     # Split data with stratification if possible
#     try:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
#     except ValueError:
#         # If stratification fails, do regular split (less ideal for imbalanced data)
#         print("Warning: Stratification failed, performing non-stratified split.")
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
    
#     # Initialize models
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
#         'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
#         'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
#     }
    
#     # Train and evaluate models
#     model_results = {}
#     trained_models = {}
    
#     for name, model in models.items():
#         print(f"\nTraining {name}...")
#         try:
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
            
#             accuracy = accuracy_score(y_test, y_pred)
#             model_results[name] = {
#                 'accuracy': accuracy,
#                 'predictions': y_pred,
#                 'classification_report': classification_report(y_test, y_pred, zero_division=0)
#             }
#             trained_models[name] = model
            
#             print(f"{name} Accuracy: {accuracy:.4f}")
#             print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
            
#         except Exception as e:
#             print(f"Error training {name}: {str(e)}")
#             continue
    
#     if not model_results:
#         print("Error: No models could be trained successfully.")
#         return None, None, None, None, None, None
    
#     # Select best model (highest accuracy)
#     best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
#     best_model = trained_models[best_model_name]
    
#     print(f"\nBest performing model: {best_model_name}")
    
#     return best_model, X_train, X_test, y_train, y_test, best_model_name

# def generate_predictions(model, customer_features, X_scaled, output_path):
#     """
#     Generate churn predictions and save to CSV
#     """
#     print("Generating predictions...")
    
#     # Make predictions
#     predictions = model.predict(X_scaled)
#     prediction_probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of churn
    
#     # Create results dataframe
#     results = customer_features[['customer_id']].copy()
#     results['actual_churn'] = customer_features['churn']
#     results['predicted_churn'] = predictions
#     results['churn_probability'] = prediction_probabilities
#     # MODIFIED RISK LEVEL BINS
#     results['risk_level'] = pd.cut(
#         prediction_probabilities,
#         bins=[0, 0.4, 0.75, 1.0], # Adjusted bins
#         labels=['Low Risk', 'Medium Risk', 'High Risk'],
#         include_lowest=True # Include lowest value in first bin
#     )
    
#     # Add key customer metrics (if available)
#     if 'transaction_date_count' in customer_features.columns:
#         results['total_transactions'] = customer_features['transaction_date_count']
#     if 'grand_total_sum' in customer_features.columns:
#         results['total_spent'] = customer_features['grand_total_sum']
#     if 'is_returned_mean' in customer_features.columns:
#         results['return_rate'] = customer_features['is_returned_mean']
#     if 'days_since_last_transaction' in customer_features.columns:
#         results['days_since_last_transaction'] = customer_features['days_since_last_transaction']
#     if 'customer_age' in customer_features.columns:
#         results['customer_age'] = customer_features['customer_age']     
    
#     # Save to CSV
#     results.to_csv(output_path, index=False)
#     print(f"Predictions saved to: {output_path}")
    
#     # Print summary statistics
#     print(f"\nPrediction Summary:")
#     print(f"Total customers: {len(results)}")
#     print(f"Predicted churners: {results['predicted_churn'].sum()}")
#     print(f"Churn rate: {results['predicted_churn'].mean():.2%}")
#     print(f"\nRisk Level Distribution:")
#     print(results['risk_level'].value_counts())
    
#     return results

# def main():
#     """
#     Main function to run the churn prediction pipeline
#     """
#     # File paths
#     input_file = "D:\Mockup\synthetic_transaction_data.csv"
#     output_file = "D:\Mockup\customer_churn_predictions.csv"
    
#     try:
#         # Load and preprocess data
#         data = load_and_preprocess_data(input_file)
        
#         # Feature engineering
#         customer_features = feature_engineering(data)
        
#         # Prepare features for modeling
#         X_scaled, y, scaler, feature_columns = prepare_features_for_modeling(customer_features)
        
#         # Train models
#         result = train_models(X_scaled, y)
#         if result[0] is None:
#             print("Model training failed. Please check your data.")
#             return
            
#         best_model, X_train, X_test, y_train, y_test, best_model_name = result
        
#         # Generate and save predictions
#         results = generate_predictions(best_model, customer_features, X_scaled, output_file)
        
#         # Feature importance (for tree-based models)
#         if hasattr(best_model, 'feature_importances_'):
#             feature_importance = pd.DataFrame({
#                 'feature': feature_columns,
#                 'importance': best_model.feature_importances_
#             }).sort_values('importance', ascending=False)
            
#             print(f"\nTop 10 Most Important Features ({best_model_name}):")
#             print(feature_importance.head(10))
        
#         print("\n" + "="*50)
#         print("CHURN PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
#         print("="*50)
        
#     except FileNotFoundError:
#         print(f"Error: Could not find the file at {input_file}")
#         print("Please ensure the file path is correct and the file exists.")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         print("Please check your data format and file permissions.")

# if __name__ == "__main__":
#     main()