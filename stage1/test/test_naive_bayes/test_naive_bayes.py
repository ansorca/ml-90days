from stage1.naive_bayes import naive_bayes as nb

def test_naive_bayes_with_numpy():
    bayes = nb.NaiveBayesClassifier()

    bayes.fit( [[1,2],[2,3],[3,1],[6,7],[7,8],[8,6]], 
              [0, 0, 0, 1, 1, 1])
    assert bayes.predict([[2,2],[7,7]]) == [0, 1]

