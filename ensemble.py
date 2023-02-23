Sadat Shahriyar
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create the individual models
sgd = SGDClassifier(random_state=42)
svm = SVC(kernel='linear', C=1.0, random_state=42)
nb = GaussianNB()

# Create the ensemble model
ensemble = VotingClassifier(estimators=[('sgd', sgd), ('svm', svm), ('nb', nb)], voting='hard')

# Fit the ensemble model to the training data
ensemble.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ensemble.predict(X_test)

# Calculate the accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble accuracy:", accuracy)