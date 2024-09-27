from mlflow.tracking import MlflowClient

def load_production_model_by_stage(model_name): 
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    current_production_version = None

    for version in versions:
            if version.tags.get("stage") == "production":
                current_production_version = version.version
    if not current_production_version : print("No available production model")

    model_uri = f"models:/{model_name}/{current_production_version}"
    
    return model_uri