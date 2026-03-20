import joblib, os, pprint
p = os.path.join(os.path.dirname(__file__), 'artifacts', 'xgb_inference_package.pkl')
if not os.path.exists(p):
    print('NO_PKL', p)
    raise SystemExit(1)
pkg = joblib.load(p)
model = pkg.get('model')
metrics = pkg.get('metrics_test')
info = {
    'run_name': pkg.get('run_name'),
    'threshold': pkg.get('threshold'),
    'metrics_test': metrics,
    'model_params': {
        'n_estimators': model.get_params().get('n_estimators'),
        'max_depth': model.get_params().get('max_depth'),
        'learning_rate': model.get_params().get('learning_rate'),
    }
}
pprint.pprint(info)
