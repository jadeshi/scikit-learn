import numpy as np

def directional_feature_importance(model,normalize=True):
    '''Computes a version of Gini importance that includes information about the direction of correlation, analogous to a beta coefficient
    in linear regression, for a single DecisionTreeRegressor. This is a slight modification of a function to calculate normal Gini importance,
    found at the following link: https://stackoverflow.com/questions/49170296/scikit-learn-feature-importance-calculation-in-decision-trees
    '''
    values = model.tree_.value.T[0][0]
    left_c = model.tree_.children_left
    right_c = model.tree_.children_right

    impurity = model.tree_.impurity    
    node_samples = model.tree_.weighted_n_node_samples 
    
    feature_importance = np.zeros((model.tree_.n_features,))

    for idx,node in enumerate(model.tree_.feature):
        if node >= 0:
            # Determine if each split in the tree corresponds to a positive or negative correlation between feature and prediction
            left_value = values[left_c[idx]]
            right_value = values[right_c[idx]]
            diff = right_value - left_value
            diff /= np.abs(diff)
            feature_importance[node]+= diff *(impurity[idx]*node_samples[idx]- \
                                   impurity[left_c[idx]]*node_samples[left_c[idx]]-\
                                   impurity[right_c[idx]]*node_samples[right_c[idx]])

    # Number of samples at the root node
    feature_importance/=node_samples[0]
    if normalize:
        feature_importance /= np.sum(np.abs(feature_importance))

    return feature_importance

def compute_ensemble_directionality(rf):
    '''Computes directional Gini importance for a RandomForestRegressor'''
    outputs = []
    for tree in rf.estimators_:
        output = direectional_feature_importance(tree)
        outputs.append(output)
    outputs = np.mean(outputs,axis=0)
    return outputs
