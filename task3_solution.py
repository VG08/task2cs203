"""
Task 3: Active Learning — Solution Code
Copy each section into the corresponding notebook cell.
"""

# ============================================================================
# CELL 1: Part 3.1 — Query Strategy Implementation
# Paste this into the cell under "Part 3.1: Query Strategy Implementation"
# ============================================================================

def least_confidence_sampling(model, X_pool, n_instances=10):
    """
    Selects samples where the model is least confident (uncertainty sampling).
    
    Args:
        model: Trained classifier with predict_proba() method
        X_pool: Feature matrix of unlabeled samples
        n_instances: Number of samples to select
    
    Returns:
        np.array: Indices of selected samples
    """
    probs = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    query_indices = np.argsort(uncertainty)[-n_instances:]
    return query_indices

def entropy_sampling(model, X_pool, n_instances=10):
    """
    Selects samples with highest entropy (information gain).
    
    Args:
        model: Trained classifier with predict_proba() method
        X_pool: Feature matrix of unlabeled samples
        n_instances: Number of samples to select
    
    Returns:
        np.array: Indices of selected samples
    """
    probs = model.predict_proba(X_pool)
    epsilon = 1e-9
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    query_indices = np.argsort(entropy)[-n_instances:]
    return query_indices

def random_sampling(model, X_pool, n_instances=10):
    """
    Baseline strategy: Selects random samples.
    
    Args:
        model: Not used, but kept for interface consistency
        X_pool: Feature matrix of unlabeled samples
        n_instances: Number of samples to select
    
    Returns:
        np.array: Randomly selected indices
    """
    n_instances = min(n_instances, len(X_pool))
    query_indices = np.random.choice(len(X_pool), size=n_instances, replace=False)
    return query_indices


# ============================================================================
# CELL 2: Part 3.2 — Data Processing and Setup
# Paste this into the cell under "Part 3.2: Data Processing and Setup"
# ============================================================================

def load_and_process_data():
    """
    Loads and processes data for active learning.
    
    Returns:
        Tuple: (X_seed, y_seed, X_pool, y_pool, X_test, y_test, vectorizer)
    """
    # Load seed (gold standard) and pool (weak labels)
    df_seed = pd.read_csv('gold_standard_reviews.csv')
    df_pool_full = pd.read_csv('weak_labels_snorkel.csv')
    
    # Use 'weak_label' column from pool; filter out Abstain rows
    df_pool_full = df_pool_full[df_pool_full['weak_label'] != 'Abstain'].reset_index(drop=True)
    
    # Map text labels to numeric: Positive=1, Negative=0, Neutral=2
    label_mapping = {
        'Positive': 1, 'positive': 1, 'POSITIVE': 1,
        'Negative': 0, 'negative': 0, 'NEGATIVE': 0,
        'Neutral': 2, 'neutral': 2, 'NEUTRAL': 2
    }
    
    # Identify label columns
    label_col_seed = 'label'
    label_col_pool = 'weak_label'
    
    # Convert seed labels
    df_seed['sentiment_numeric'] = df_seed[label_col_seed].map(label_mapping)
    
    # Convert pool labels
    df_pool_full['sentiment_numeric'] = df_pool_full[label_col_pool].map(label_mapping)
    
    # Create static test set (hold out 50 samples from pool)
    df_pool, df_test = train_test_split(df_pool_full, test_size=50, random_state=42)
    
    # Vectorize text data using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    all_text = pd.concat([df_seed['review'], df_pool['review'], df_test['review']])
    vectorizer.fit(all_text)
    
    # Transform datasets to feature matrices
    X_seed = vectorizer.transform(df_seed['review']).toarray()
    X_pool = vectorizer.transform(df_pool['review']).toarray()
    X_test = vectorizer.transform(df_test['review']).toarray()
    
    # Extract numeric labels
    y_seed = df_seed['sentiment_numeric'].values
    y_pool = df_pool['sentiment_numeric'].values
    y_test = df_test['sentiment_numeric'].values
    
    return X_seed, y_seed, X_pool, y_pool, X_test, y_test, vectorizer

X_seed, y_seed, X_pool, y_pool, X_test, y_test, vectorizer = load_and_process_data()

print(f"Seed Size: {len(y_seed)}")
print(f"Pool Size: {len(y_pool)} (Available for querying)")
print(f"Test Size: {len(y_test)} (Held out for evaluation)")


# ============================================================================
# CELL 3: Part 3.3 — Active Learning Loop
# Paste this into the cell under "Part 3.3: Active Learning Loop"
# ============================================================================

def run_active_learning_loop(X_seed, y_seed, X_pool, y_pool, X_test, y_test,
                             strategy_func, steps=5, batch_size=10):
    """
    Simulates the active learning loop.
    
    Args:
        X_seed, y_seed: Initial training data (seed set)
        X_pool, y_pool: Unlabeled pool (y_pool is hidden, revealed during query)
        X_test, y_test: Static test set for evaluation
        strategy_func: Function that selects samples
        steps: Number of iterations
        batch_size: Number of samples to query per iteration
    
    Returns:
        Tuple: (n_labels_history, accuracy_history)
    """
    # Initialize training set with seed data
    X_train = X_seed.copy()
    y_train = y_seed.copy()
    
    # Create working copies of pool
    X_pool_curr = X_pool.copy()
    y_pool_curr = y_pool.copy()
    
    # Initialize tracking lists
    accuracy_history = []
    n_labels_history = []
    
    # Train initial model on seed data
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate initial model
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracy_history.append(acc)
    n_labels_history.append(len(y_train))
    print(f"Initial — Labels: {len(y_train)}, Test Accuracy: {acc:.4f}")
    
    # Iterative active learning loop
    for i in range(steps):
        if len(X_pool_curr) == 0:
            print(f"Pool exhausted at step {i}")
            break
        
        # Adjust batch size if pool is smaller
        actual_batch = min(batch_size, len(X_pool_curr))
        
        # 1. Query: select most informative samples
        query_indices = strategy_func(model, X_pool_curr, n_instances=actual_batch)
        
        # 2. "Label": reveal ground truth
        X_new = X_pool_curr[query_indices]
        y_new = y_pool_curr[query_indices]
        
        # 3. Add to training set
        X_train = np.vstack([X_train, X_new])
        y_train = np.concatenate([y_train, y_new])
        
        # 4. Remove from pool
        X_pool_curr = np.delete(X_pool_curr, query_indices, axis=0)
        y_pool_curr = np.delete(y_pool_curr, query_indices, axis=0)
        
        # 5. Retrain model
        model.fit(X_train, y_train)
        
        # 6. Evaluate
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy_history.append(acc)
        n_labels_history.append(len(y_train))
        print(f"Step {i+1} — Labels: {len(y_train)}, Test Accuracy: {acc:.4f}")
    
    return n_labels_history, accuracy_history

# Run active learning with least confidence strategy
print("=== Least Confidence Sampling ===")
n_labels_lc, acc_lc = run_active_learning_loop(
    X_seed, y_seed, X_pool, y_pool, X_test, y_test,
    strategy_func=least_confidence_sampling, steps=5, batch_size=10
)

print("\n=== Entropy Sampling ===")
n_labels_ent, acc_ent = run_active_learning_loop(
    X_seed, y_seed, X_pool, y_pool, X_test, y_test,
    strategy_func=entropy_sampling, steps=5, batch_size=10
)


# ============================================================================
# CELL 4: Part 3.4 — Visualization and Comparison
# Paste this into the cell under "Part 3.4: Visualization and Comparison"
# ============================================================================

# Run active learning with random sampling (baseline)
print("=== Random Sampling (Baseline) ===")
np.random.seed(42)
n_labels_rand, acc_rand = run_active_learning_loop(
    X_seed, y_seed, X_pool, y_pool, X_test, y_test,
    strategy_func=random_sampling, steps=5, batch_size=10
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(n_labels_lc, acc_lc, 'b-o', label='Least Confidence', linewidth=2)
plt.plot(n_labels_ent, acc_ent, 'g-s', label='Entropy Sampling', linewidth=2)
plt.plot(n_labels_rand, acc_rand, 'r--^', label='Random Sampling', linewidth=2)
plt.xlabel('Number of Labeled Samples', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Active Learning: Learning Curves Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print comparison summary
print("\n=== Final Accuracy Comparison ===")
print(f"Least Confidence: {acc_lc[-1]:.4f} (with {n_labels_lc[-1]} labels)")
print(f"Entropy Sampling: {acc_ent[-1]:.4f} (with {n_labels_ent[-1]} labels)")
print(f"Random Sampling:  {acc_rand[-1]:.4f} (with {n_labels_rand[-1]} labels)")
