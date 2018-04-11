
class Stacker():
    import numpy as np
    from sklearn.model_selection import KFold

    def __init__(self, X, y, cv=5, seed=1):
        self.n_folds=cv
        self.seed = seed
        self.X_orig = X
        self.y_orig = y
        
    def _sample_split(self, X, y, train_index, test_index):
        '''
        Input:  Feature set X(n,p) and y(n) with indices for train / test splits
        
        Return: Feature sets X_train, X_test and responses y_train, y_test according to split input
        '''
        try:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        except:
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]
        
        return X_train, y_train, X_test, y_test
    
    
    def _fit_predict(self, model, X_train, y_train, X_test):
        '''
        Input:  Single model estimator to be trained on X_train(n,p) and y_train(n,),
                along with feature set to be predicted, X_test(m,p)
                
        Return: Prediction results of shape (m,) for the input X_test
        '''
        model.fit(X_train, y_train)
        single_prediction = model.predict(X_test)
        
        return single_prediction
    
    
    def _cross_val(self, model, X, y, folds):
        '''
        Input:  Single model estimator with X(n,p) and y(n,) datasets to be sequentially
                trained and predicted with an entire k-fold process.
                
        Return: Prediction results of shape (n,) aggregated together 
                from each out-of-fold (oof) prediction
        '''        
        oof_predictions = self.np.zeros(shape=(len(X),))
        kf = self.KFold(n_splits=folds, random_state=self.seed)
            
        for train_index, test_index in kf.split(X):
            X_train, y_train, X_test, y_test = self._sample_split(X, y, train_index, test_index)
            oof_prediction_n = self._fit_predict(model, X_train, y_train, X_test)
            oof_predictions[test_index] = oof_prediction_n
                
        return oof_predictions

    
    def _multiple_model_fit_predict(self, X_train, y_train, X_test):
        '''
        Input:  Training datasets X_train(n,p) and y_train(n,), along with 
                feature set to be predicted, X_test(m,p). Also inherits the 
                self.estimator attribute for various input estimators.
                
        Return: Meta feature set X_stack of shape (m,len(e)) where len(e) 
                is the number of estimators
        '''
        X_stack = self.np.zeros(shape=(len(X_test), self.num_stack_features))
        
        for j, model in enumerate(self.estimators):
            test_predictions = self._fit_predict(model, X_train, y_train, X_test)
            X_stack[:,j] = test_predictions
        
        return X_stack
    
    
    def _multiple_model_cross_val(self, X_train, y_train, folds):
        '''
        Input:  Training datasets X_train(n,p) and y_train(n,), along with 
                feature set to be predicted, X_test(m,p). Also inherits the 
                self.estimator attribute for various input estimators.
                
        Return: Meta feature set X_stack of shape (m,len(e)) where m is aggregated
                predictions from cross-validation and len(e) is the number of estimators
        '''
        for j, model in enumerate(self.estimators):
            try:
                print('Training ' + self.names[j] + ' Fold ' + str(self.i+1) + '/' + str(self.n_folds))
            except: 
                pass
            
            test_predictions = self._cross_val(model, X_train, y_train, folds)
            
            if j == 0:
                X_stack = self.np.zeros(shape=(len(test_predictions), self.num_stack_features))
            else: 
                pass
            
            X_stack[:,j] = test_predictions
        
        return X_stack
        
        
    def layer_one_cv(self, estimators, names):
        '''
        Input:  Full feature set X(n,p) and y(n) with a list of estimators (and their names)
                to use for training the first layer of the stacking ensemble.
                
        Return: None, but it does create an inheritable dictionary that contains meta stacked
                feature sets: X_stack(n*(k-1)/k, len(e)) and X_holdout(n*1/k, len(e)) for each
                of k folds and number of estimators len(e). These can be passed the second
                layer for training and validating the stacker estimator.
        '''
        self.estimators = estimators
        self.names = names
        self.num_stack_features = len(estimators)

        self.layer_one_results = {}    # dictionary to store inner folds and holdout stacking datasets

        kf = self.KFold(n_splits=self.n_folds, random_state=self.seed)
        
        for i, (inside_index, holdout_index) in enumerate(kf.split(self.X_orig)):
            self.i = i
            X_folds, y_folds, X_holdout, y_holdout = self._sample_split(self.X_orig, self.y_orig, 
                                                                        inside_index, holdout_index)
            
            # creating arrays to hold inner fold and holdout results
            self.layer_one_results['fold_' + str(i+1)] = {}
                
            X_stack_folds = self._multiple_model_cross_val(X_folds, y_folds, folds=self.n_folds-1)
            X_stack_holdout = self._multiple_model_fit_predict(X_folds, y_folds, X_holdout)
            
            self.layer_one_results['fold_' + str(i+1)]['X_stack_folds'] = X_stack_folds
            self.layer_one_results['fold_' + str(i+1)]['y_folds'] = y_folds
            self.layer_one_results['fold_' + str(i+1)]['X_stack_holdout'] = X_stack_holdout
            self.layer_one_results['fold_' + str(i+1)]['y_holdout'] = y_holdout
    
    
    def layer_two_cv(self, stack_estimator):
        '''
        Input:  Stacking estimator. Also inherits the layer_one dictionary that contains
                the meta feature sets for X_stack_folds(n*(k-1)/k, len(e)) and 
                X_stack_holdout(n*1/k, len(e)), as well as corresponding targets y_stack_folds
                and y_stack_holdout
        
        Return: 
        '''
        from sklearn.metrics import mean_squared_error
        
        layer_two_errors = {}
        
        for i, fold in enumerate(self.layer_one_results.items()):
            X_stack_folds = fold[1]['X_stack_folds']
            y_stack_folds = fold[1]['y_folds']
            X_stack_holdout = fold[1]['X_stack_holdout']
            y_stack_holdout = fold[1]['y_holdout']
            
            predictions_two = self._fit_predict(stack_estimator, 
                                                X_stack_folds, y_stack_folds, 
                                                X_stack_holdout)
            
            layer_two_errors['fold_' + str(i+1)] = self.np.sqrt(mean_squared_error(predictions_two, y_stack_holdout))
            
        return layer_two_errors
    
    
    def layer_one_results(self):
        return self.layer_one_results
    
    
    def _final_model_train_stacker(self, stack_estimator):
                
        X_stack_cv = self._multiple_model_cross_val(self.X_orig, self.y_orig, folds=self.n_folds)
        trained_stacker = stack_estimator.fit(X_stack_cv, self.y_orig)
        
        return trained_stacker
        
        
    def final_model_train(self, estimators, stack_estimator):
        
        self.estimators = estimators
        
        self.trained_stacker = self._final_model_train_stacker(stack_estimator)
        
        self.trained_models = []
        for model in self.estimators:
            model.fit(self.X_orig, self.y_orig)
            self.trained_models.append(model)
            
            
    def final_model_predict(self, X_test):
        
        X_test_stack = self.np.zeros(shape=(len(X_test), len(self.trained_models)))
        
        for j, model in enumerate(self.trained_models):
            results = model.predict(X_test)
            X_test_stack[:,j] = results
            
        return self.trained_stacker.predict(X_test_stack)