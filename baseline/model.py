from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

def choose_base_model(X_train, y_train, args):
    # Define Bayesian search spaces for different model types
    param_spaces = {
        'SVR': {
            'C': Real(0.1, 1000, prior='log-uniform'),
            'gamma': Real(1e-4, 1, prior='log-uniform'),
            'epsilon': Real(0.01, 1, prior='uniform'),
            'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': Integer(2, 3),
            'coef0': Real(-0.5, 0.5)
        },
        'RF': {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        },
        'GBR': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'max_depth': Integer(3, 15),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'subsample': Real(0.6, 1.0, prior='uniform')
        }
    }
    
    # Define model instances for individual model types
    model_dict = {
       'SVR': SVR(),
       'RF': RandomForestRegressor(random_state=args.seed),
       'GBR': GradientBoostingRegressor(random_state=args.seed)
    }
    
    if args.method == 'ensemble':
        # Optimize each base model with BayesSearchCV
        svr_search = BayesSearchCV(
            estimator=SVR(),
            search_spaces=param_spaces['SVR'],
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            random_state=args.seed
        )
        rf_search = BayesSearchCV(
            estimator=RandomForestRegressor(random_state=args.seed),
            search_spaces=param_spaces['RF'],
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            random_state=args.seed
        )
        gbr_search = BayesSearchCV(
            estimator=GradientBoostingRegressor(random_state=args.seed),
            search_spaces=param_spaces['GBR'],
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            random_state=args.seed
        )
        
        # Fit the BayesSearchCV objects
        svr_search.fit(X_train, y_train)
        rf_search.fit(X_train, y_train)
        gbr_search.fit(X_train, y_train)
        
        # Retrieve the best estimators for each base model
        best_svr = svr_search.best_estimator_
        best_rf = rf_search.best_estimator_
        best_gbr = gbr_search.best_estimator_
        
        # Build the stacking ensemble using the optimized base models
        base_models = [
            ('SVR', best_svr),
            ('RF', best_rf),
            ('GBR', best_gbr)
        ]
        meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
        ensemble = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=args.cv)
        ensemble.fit(X_train, y_train)
        
        # Directly return the fitted ensemble for prediction
        return ensemble
    
    else:
        # For individual model types
        if args.method not in param_spaces:
            raise ValueError(f"Model type '{args.method}' not supported.")
        
        model = model_dict[args.method]
        param_space = param_spaces[args.method]
        
        search_cv = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            random_state=args.seed
        )
        
        search_cv.fit(X_train, y_train)
        # Return the best estimator directly for prediction
        return search_cv.best_estimator_
