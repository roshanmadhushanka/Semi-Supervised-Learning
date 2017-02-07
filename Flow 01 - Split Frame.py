import h2o

h2o.init()

data = h2o.import_file('iris.csv')
train, test, validate = data.split_frame(ratios=[0.1, 0.8])

h2o.export_file(frame=train, force=True, path='train.csv')
h2o.export_file(frame=test, force=True, path='test.csv')
h2o.export_file(frame=validate, force=True, path='validate.csv')
