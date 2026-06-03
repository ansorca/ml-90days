from stage1.knn import knn_with_numpy

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