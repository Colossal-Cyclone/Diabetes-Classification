import pandas as pd

data=pd.read_csv('pima-indians-diabetes.csv')

cols= ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

data[cols]= data[cols].apply(lambda x: (x-x.min())/(x.max()-x.min()))

import tensorflow as tf

nm =tf.feature_column.numeric_column('Number_pregnant')
gc =tf.feature_column.numeric_column('Glucose_concentration')
bp =tf.feature_column.numeric_column('Blood_pressure')
tr =tf.feature_column.numeric_column('Triceps')
iu =tf.feature_column.numeric_column('Insulin')
bm =tf.feature_column.numeric_column('BMI')
pe =tf.feature_column.numeric_column('Pedigree')
ag =tf.feature_column.numeric_column('Age')

assg = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=1000)

emg = tf.feature_column.embedding_column(assg, dimension=4)

ag_b=tf.feature_column.bucketized_column(ag , boundaries=[10,20,30,40,50,60,70,80])

feat_cols= [nm,gc,bp,tr,iu,bm,ag_b,emg]

x= data.drop('Class', axis=1)

labels=data['Class']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, labels, test_size=0.33, random_state=100)

input_func=tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=18, num_epochs=5000, shuffle=True)

model=tf.estimator.DNNClassifier(hidden_units=[9,9,9,3], feature_columns=feat_cols, n_classes=2)

model.train(input_fn=input_func, steps=25000)

eval_func=tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test, batch_size=8, num_epochs=1, shuffle=False)

results=model.evaluate(input_fn=eval_func)

print(results)