from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x = iris.data.features
y = iris.data.targets

# metadata
#print(iris.metadata)

# variable information
#print(iris.variables.values)

print(iris.data.features)

##print(y)

def iris_data():
    return iris.data
