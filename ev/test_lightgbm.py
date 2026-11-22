try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    print("✓ LightGBM imported successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
