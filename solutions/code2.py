spam_ET = ExtraTreesClassifier(n_estimators=200, criterion='entropy')
spam_ET.fit(Xtrain,ytrain)

spam_ET.score(Xtest.toarray(),ytest)

# Compute cross-validation score
nb_trials = 20
score = []
for i in range(nb_trials):
    Xtrain, ytrain, Xtest, ytest = spam_data.shuffle_and_split(2000, feat='wordcount')
    spam_ET = ExtraTreesClassifier(n_estimators=200, criterion='entropy')
    spam_ET.fit(Xtrain,ytrain);
    score += [spam_ET.score(Xtest,ytest)]
    print('*', end='')
print(" done!")

print("Average generalization score:", np.mean(score))
print("Standard deviation:", np.std(score))
