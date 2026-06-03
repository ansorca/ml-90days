# stand-alone code 
from sklearn import (datasets, metrics,  model_selection as skms, naive_bayes, neighbors)  
# we set random_state so the results are reproducible 
# # otherwise, we get different training and testing sets 
# # more details in Chapter 5 

iris = datasets.load_iris()

(iris_train_ftrs, iris_test_ftrs, iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,  
                                                                                         iris.target, 
                                                                                         test_size=.90, 
                                                                                         random_state=42) 
models = {'kNN': neighbors.KNeighborsClassifier(n_neighbors=3),
          'NB' : naive_bayes.GaussianNB()} 
          
for name, model in models.items():
    fit = model.fit(iris_train_ftrs, iris_train_tgt) 
    predictions = fit.predict(iris_test_ftrs)  
    score = metrics.accuracy_score(iris_test_tgt, predictions) 
    print("{:>3s}: {:0.2f}".format(name,score))