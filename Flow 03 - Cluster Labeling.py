import h2o
import pandas as pd


# Please enter the model name
model_path = 'KMeans_model_python_1484507298817_1'


def calculate_label_value(labels):
    label_value = {}
    for label in labels:
        if label_value.has_key(label):
            label_value[label] += 1
        else:
            label_value[label] = 1

    # Balancing
    for key in label_value.keys():
        label_value[key] = 1.0 / label_value[key]

    return label_value


def generate_cluster_data(model_path, input_data, response_data):
    h2o.init()

    input_data = pd.DataFrame(input_data)
    input_frame = h2o.H2OFrame(input_data)
    response_data = pd.DataFrame(response_data)

    cluster_data = pd.DataFrame()

    model = h2o.load_model(model_path)
    predictions = model.predict(test_data=input_frame)
    predictions = predictions.as_data_frame(use_pandas=True)

    cluster_data['cluster_index'] = predictions
    cluster_data['cluster_label'] = response_data

    return cluster_data


def label_cluster(cluster_data, label_values):
    label_values = dict(label_values)
    cluster_data = pd.DataFrame(cluster_data)

    cluster_vote = {}
    for i in range(len(cluster_data.index)):
        index = cluster_data.iloc[i, 0]
        name = cluster_data.iloc[i, 1]

        if cluster_vote.has_key(index):
            if cluster_vote[index].has_key(name):
                cluster_vote[index][name] += label_values[name]
            else:
                cluster_vote[index][name] = label_values[name]
        else:
            cluster_vote[index] = {}

    labeled = [] # Contains all the used labels
    for index in cluster_vote.keys():
        max_vote = 0.0
        max_label = None
        for label in cluster_vote[index].keys():
            if label in labeled:
                continue
            current_vote = cluster_vote[index][label]
            if current_vote > max_vote:
                max_vote = current_vote
                max_label = label
        cluster_vote[index] = max_label
        labeled.append(max_label)

    return cluster_vote


response_column = 'class'
validation_frame = pd.read_csv('validate.csv')
response_column_data = validation_frame[response_column]

# Calculate label values
label_values = calculate_label_value(response_column_data)

# Generate cluster data
input_data = pd.read_csv('validate.csv')
response_data = validation_frame[response_column]
del input_data[response_column]
cluster_data = generate_cluster_data(model_path=model_path, input_data=input_data, response_data=response_data)

# Label cluster
cluster_labels = label_cluster(cluster_data, label_values)
print cluster_labels

h2o.init()
input_data = pd.read_csv('test.csv')
response_data = list(input_data[response_column])
del input_data[response_column]
input_frame = h2o.H2OFrame(input_data)

model = h2o.load_model(model_path)
predictions = model.predict(test_data=input_frame)
h2o.export_file(frame=predictions, path='prediction.csv', force=True)
predictions = list(predictions.as_data_frame(use_pandas=True)['predict'])

for i in range(len(predictions)):
    predictions[i] = cluster_labels[predictions[i]]

match_count = 0
for i in range(len(predictions)):
    if predictions[i] == response_data[i]:
        match_count += 1
    else:
        print 'actual :', response_data[i], ' preditc :', predictions[i]

print 'match', match_count
print 'mismatch', len(predictions) - match_count





