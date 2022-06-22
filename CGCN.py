import numpy as np
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, concatenate, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from stellargraph.layer import GCN
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from utils.MyLabelBinarizer import *
from utils.MyEvaluate import *
from IPython.display import display, HTML
import tensorflow as tf

# Set random seed
seed = 123
tf.random.set_seed(seed)

######################################################################################################################
baseURL = "./downstream_task/data/"
######################################################################################################################

def TextCNN_model(main_input, embedding_matrix, project):
    main_input = tf.squeeze(main_input, axis=0)
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                         output_dim=embedding_matrix.shape[1],
                         input_length=get_avg_len(project),
                         weights=[embedding_matrix],
                         trainable=False)
    embed = embedding(main_input)
    cnn = Conv1D(filters=10, kernel_size=5, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPooling1D(pool_size=int(cnn.shape[1]))(cnn)
    hidden = Dense(32, activation='relu')(cnn)
    flat = Flatten()(hidden)
    drop = Dropout(0.1)(flat)
    output = tf.expand_dims(flat, axis=0)
    model = Model(inputs=main_input, outputs=output)
    return model

def load_embedding_matrix(path):
    embed_matrix_file = pd.read_csv(path, header=0, index_col=False)
    embed_matrix = np.array(embed_matrix_file.iloc[:, 1:])
    return embed_matrix

def get_avg_len(project):
    tokens_integer = []
    len_list = []
    tokens_integer_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\tokens_map.txt", 'r')
    # tokens_integer_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\tokens_map_cross.txt", 'r')
    lines = tokens_integer_file.readlines()
    for each_line in lines:
        integer = each_line[each_line.index('\t') + 1:].strip('\n')
        integer_list = integer.split(' ')
        tokens_integer.append(integer_list[:-1])
        len_list.append(len(integer_list[:-1]))
    max_len = max(len_list)
    return max_len

def sequence(tokens_seq,project):
    return pad_sequences(tokens_seq, maxlen=get_avg_len(project),padding='post',truncating='post')

def load_CNN_data(project, train_index, test_index):
    tokens_map = []
    tokens_map_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\tokens_map.txt", 'r')
    # tokens_map_file = open(baseURL + project.split('-')[0] + "\\" + project + "\\tokens_map_cross.txt", 'r')
    lines = tokens_map_file.readlines()
    for each_line in lines:
        integer = each_line[each_line.index('\t') + 1:].strip('\n')
        integer_list = integer.split(' ')
        tokens_map.append(integer_list[:-1])

    process = lambda data,label: (sequence(data,project), to_categorical(label))
    origin_data = pd.read_csv(baseURL + project.split('-')[0] + "\\" + project + "\\Process-Binary.csv",header=0,index_col=False)
    Alldata = process(tokens_map,origin_data['bug'])

    X = Alldata[0]
    y = np.argmax(Alldata[1], axis=1)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test, Alldata

def load_GCN_data(project):
    if project == "ant-1.4":
        dataset = sg.datasets.ant_1_4()
    elif project == "ant-1.6":
        dataset = sg.datasets.ant_1_6()
    elif project == "ant-1.7":
        dataset = sg.datasets.ant_1_7()
    elif project == "camel-1.2":
        dataset = sg.datasets.camel_1_2()
    elif project == "camel-1.4":
        dataset = sg.datasets.camel_1_4()
    elif project == "camel-1.6":
        dataset = sg.datasets.camel_1_6()
    elif project == "jedit-3.2":
        dataset = sg.datasets.jedit_3_2()
    elif project == "jedit-4.0":
        dataset = sg.datasets.jedit_4_0()
    elif project == "jedit-4.1":
        dataset = sg.datasets.jedit_4_1()
    elif project == "lucene-2.0":
        dataset = sg.datasets.lucene_2_0()
    elif project == "lucene-2.2":
        dataset = sg.datasets.lucene_2_2()
    elif project == "lucene-2.4":
        dataset = sg.datasets.lucene_2_4()
    elif project == "poi-1.5":
        dataset = sg.datasets.poi_1_5()
    elif project == "poi-2.5":
        dataset = sg.datasets.poi_2_5()
    elif project == "poi-3.0":
        dataset = sg.datasets.poi_3_0()
    elif project == "velocity-1.4":
        dataset = sg.datasets.velocity_1_4()
    elif project == "velocity-1.5":
        dataset = sg.datasets.velocity_1_5()
    elif project == "velocity-1.6":
        dataset = sg.datasets.velocity_1_6()
    elif project == "xalan-2.4":
        dataset = sg.datasets.xalan_2_4()
    elif project == "xalan-2.5":
        dataset = sg.datasets.xalan_2_5()
    elif project == "xalan-2.6":
        dataset = sg.datasets.xalan_2_6()

    display(HTML(dataset.description))
    G, node_subjects = dataset.load()
    print(G.info())
    print(node_subjects.value_counts().to_frame())

    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.8, test_size=None, stratify=node_subjects, random_state=0
    )

    generator = FullBatchNodeGenerator(G, method="gcn")

    target_encoding = MyLabelBinarizer()
    node_targets = target_encoding.fit_transform(node_subjects)
    train_targets = target_encoding.fit_transform(train_subjects)
    test_targets = target_encoding.fit_transform(test_subjects)

    gen = generator.flow(node_subjects.index, node_targets)
    train_gen = generator.flow(train_subjects.index, train_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    gcn = GCN(
        layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.0
    )
    x_inp, x_out = gcn.in_out_tensors()

    return x_inp, x_out, gen, train_gen, test_gen, train_subjects, test_subjects

def mix(inp1, inp2, r1, r2, Alldata, X_train, X_test, gen, train_gen, test_gen, project):
    alpha = 0.5
    # Concatenate
    # concat = tf.keras.layers.Concatenate(axis=2)([(1-alpha)*r1, alpha*r2])
    # concat = tf.keras.layers.Concatenate(axis=2)([r1, r2])
    # add
    # concat = tf.keras.layers.Add()([r1, r2])
    concat = tf.keras.layers.Add()([(1-alpha)*r1, alpha*r2])
    # Multiply
    # concat = tf.keras.layers.Multiply()([r1, r2])
    dense1 = Dense(32, activation='relu')(concat)
    prediction = Dense(2, activation='softmax', name="pred_Layer")(dense1)

    merged_model = Model(inputs=[inp1, inp2], outputs=prediction)

    merged_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=['binary_crossentropy'],
                         metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=50,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    data_in = np.expand_dims(Alldata[0], axis=0)
    train_in = np.expand_dims(X_train, axis=0)
    test_in = np.expand_dims(X_test, axis=0)
    history = merged_model.fit(x=[train_in, train_gen.inputs], y=train_gen.targets,
                               epochs=500, verbose=2,
                               validation_data=([test_in, test_gen.inputs], test_gen.targets),
                               callbacks=[learning_rate_reduction]
                               )

    merged_model.evaluate(x=[test_in, test_gen.inputs], y=test_gen.targets, verbose=2)

    test_predictions = merged_model.predict([test_in, test_gen.inputs])
    test_predictions = test_predictions.squeeze(0)
    test_pred = [np.argmax(one_hot) for one_hot in test_predictions]
    labels = test_gen.targets.squeeze(0)
    test_label = [np.argmax(one_hot) for one_hot in labels]

    metric(test_label, test_pred)
    test_auc = metrics.roc_auc_score(test_label, test_pred)
    print(project + " AUC：", test_auc)

    embedding_model = tf.keras.Model(inputs=[inp1, inp2], outputs=dense1)
    emb = embedding_model.predict([data_in, gen.inputs])
    X = emb.squeeze(0)
    # save embeddings
    df = pd.DataFrame(X, columns=[('emb_' + str(i)) for i in range(X.shape[1])])
    df.to_csv(baseURL + project.split('-')[0] + "\\" + project + '\\CGCN_emb_add_alpha=0.5.csv', index=False)


if __name__ == '__main__':
    projects = ["ant-1.4", "ant-1.6", "ant-1.7", "camel-1.2", "camel-1.4", "camel-1.6",
                "jedit-3.2", "jedit-4.0", "jedit-4.1", "lucene-2.0", "lucene-2.2", "lucene-2.4",
                "poi-1.5", "poi-2.5", "poi-3.0", "velocity-1.4", "velocity-1.5", "velocity-1.6",
                "xalan-2.4", "xalan-2.5", "xalan-2.6"]

    # cross project
    # projects = ["ant-1.4", "camel-1.2", "jedit-3.2", "lucene-2.0", "poi-1.5", "velocity-1.4", "xalan-2.4"]

    for i in range(len(projects)):
        # load GCN data
        x_inp, x_out, gen, train_gen, test_gen, train_subjects, test_subjects = load_GCN_data(projects[i])

        embedding_matrix = load_embedding_matrix(
            baseURL + projects[i].split('-')[0] + "\\" + projects[i] + "\\vocab_emb_dict_30.csv")
        # load CNN data
        X_train, y_train, X_test, y_test, Alldata = load_CNN_data(projects[i], train_subjects.index, test_subjects.index)
        inpCNN = Input(batch_shape=(1, None, get_avg_len(projects[i])), dtype='float64', name='cnn_input')
        modelCNN = TextCNN_model(inpCNN, embedding_matrix, projects[i])
        r1 = modelCNN.output
        r2 = x_out
        mix(inpCNN, x_inp, r1, r2, Alldata, X_train, X_test, gen, train_gen, test_gen, projects[i])
