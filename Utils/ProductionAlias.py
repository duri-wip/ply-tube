from mlflow.tracking import MlflowClient

def production_alias(model_name, param):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    best_model_version = None
    best_param = 0
    current_production_version = None

    # 먼저 production alias가 이미 지정된 모델을 찾기 (tags 사용)
    for version in versions:
        if version.tags.get("stage") == "production":
            current_production_version = version.version
    print(current_production_version)

    # 최고 성능의 모델 찾기
    for version in versions:
        run_id = version.run_id
        param_value = client.get_run(run_id).data.metrics.get(f'{param}',0)

        if param_value > best_param:
            best_param = param_value
            best_model_version = version.version

    # 이전 production alias를 삭제하고 새로 지정하기
    if best_model_version:
        # 기존 production alias가 있는 경우, 그 alias 삭제
        if current_production_version and current_production_version != best_model_version:
            client.delete_model_version_tag(
                name=model_name,
                version=current_production_version,
                key="stage"
            )
            print(f"Previous production alias removed from version {current_production_version}")

        # 새로 선택된 모델에 production alias 지정
        client.set_model_version_tag(
            name=model_name,
            version=best_model_version,
            key="stage",
            value="production"
        )
        print(f"Model version {best_model_version} with {param} = {best_param} set to 'production' alias.")
    else:
        print(f"No models found or no model with {param} metric found.")
