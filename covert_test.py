from keras_pandas.Automater import Automater
from keras_pandas.lib import load_titanic


csv_file = 'energy_data.csv'
observations = csv_as_belief(csv_file,-9,92)

# Transform the data set, using keras_pandas
numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']
text_vars = ['name']

auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, text_vars=text_vars,
 response_var='survived')
X, y = auto.fit_transform(observations)
