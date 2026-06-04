from stage1.knn import knn_with_numpy
import pytest

def test_knn_with_numpy():
    knn = knn_with_numpy.KNN_numpy(5)
    knn.fit( [[1,2],[2,3],[3,1],[6,7],[7,8],[8,6]], [0, 0, 0, 1, 1, 1])
    assert knn.predict([[2,2],[7,7]]) == [0, 1]

def test_knn_with_numpy_irregular_data():
    X_train = [[1,1],[1,2],[2,1],[2,2],[5,5],[8,8],[9,8],[8,9]]
    y_train = [  0,    0,    0,    1,    0,    1,    1,    1  ]
    X_test  = [[3,3]]

    results =[]
    for k in range(1,4,2):
        knn = knn_with_numpy.KNN_numpy(k)
        knn.fit(X_train, y_train)
        results.append(knn.predict(X_test))

    assert results == [[1], [0]]

def test_knn_k_larger_than_training_set():
    knn = knn_with_numpy.KNN_numpy(k=10)
    knn.fit([[1,2],[2,3]], [0, 1])
    # should not crash — what should it do?
    result = knn.predict([[1,2]])

def test_knn_predict_without_fit():
    knn = knn_with_numpy.KNN_numpy(k=3)
    with pytest.raises(Exception):
        knn.predict([[1,2]])