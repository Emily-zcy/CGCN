# Software defect prediction with semantic and structural information of codes based on Graph Neural Networks

Supplementary code and data of the paper *Software defect prediction with semantic and structural information of codes based on Graph Neural Networks*.

@article{zhou2022software,
  title={Software defect prediction with semantic and structural information of codes based on Graph Neural Networks},
  author={Zhou, Chunying and He, Peng and Zeng, Cheng and Ma, Ju},
  journal={Information and Software Technology},
  volume={152},
  pages={107057},
  year={2022},
  publisher={Elsevier}
}

This work is the extension of *GCN2defect: Graph Convolutional Networks for SMOTETomek-based Software Defect Prediction*.

@INPROCEEDINGS{9700305, 
author={Zeng, Cheng and Zhou, Chun Ying and Lv, Sheng Kai and He, Peng and Huang, Jie}, 
booktitle={2021 IEEE 32nd International Symposium on Software Reliability Engineering (ISSRE)}, title={GCN2defect : Graph Convolutional Networks for SMOTETomek-based Software Defect Prediction}, 
year={2021}, 
volume={}, 
number={}, 
pages={69-79}, 
doi={10.1109/ISSRE52982.2021.00020}}

### Generating Class Dependency Network

---

In each subdirectory, there is already exsit the corresponding Class Dependency Network (CDN) (node.txt and edges.txt). If you want to generate your own CDN, you can use the *Dependencyfinder API*.

### Generating AST

------

We have placed the processed AST and extracted token sequences (tokens.txt) in each subdirectory.

### Generating the initial node attributes

---

Before  training  CGCN,  we have to provide the attributes of the CDN nodes. Thus, three types of node metrics are introuduced as node attributes:

*1) Traditional  Static  Code  Metric:* 20 manually designed metrics (Process-Binary.csv).

*2) Network  Embedding  Metric:* use the [ProNE](https://github.com/THUDM/ProNE) implementation to generate the network embedding file.

### Generating the CGCN embeddings

---

Run CGCN.py to generate embeddings.

The GCN part of our model is modified based on stellargraph](https://github.com/stellargraph/stellargraph). The GCN demo shows in https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html. 

If you want to change to your own dataset, you need the following steps:

**1) Replace the name in the red box in the following figure with the name of your dataset.**

 ![image-20220305213519599](https://github.com/Emily-zcy/GCN2defect/blob/main/img/image-20220305213519599.png)

**2) Place the mouse over the *dataset*, then press Ctrl, and click to enter *_init_.py*.**

 ![image-20220305213816599](https://github.com/Emily-zcy/GCN2defect/blob/main/img/image-20220305213816599.png)

**Add the name of your dataset in *_init_.py*.** 

 ![image-20220305214236889](https://github.com/Emily-zcy/GCN2defect/blob/main/img/image-20220305214236889.png)

**3) Place the mouse over the dataset name (except for the dataset name you just created), then press Ctrl, and click to enter *datasets.py*.**

 ![image-20220305214527747](https://github.com/Emily-zcy/GCN2defect/blob/main/img/image-20220305214527747.png)

**4) Create your own class in *datasets.py*. For example, the following code is to create Ant dataset:**

```python
class Ant(
    DatasetLoader,
    name="Ant",
    directory_name="Ant",
    url="",
    url_archive_format="",
    expected_files=[],
    description="",
    source="",
):
    _NUM_FEATURES = 20
    def load(
        self,
        directed=False,
        largest_connected_component_only=False,
        subject_as_feature=False,
        edge_weights=None,
        str_node_ids=False,
    ):
        nodes_dtype = str if str_node_ids else int

        return _load_defect_data(
            self,
            directed,
            largest_connected_component_only,
            subject_as_feature,
            edge_weights,
            nodes_dtype,
        )
```

```python
def _load_defect_data(
    dataset,
    directed,
    largest_connected_component_only,
    subject_as_feature,
    edge_weights,
    nodes_dtype,
):
    assert isinstance(dataset, (Ant))
    if nodes_dtype is None:
        nodes_dtype = dataset._NODES_DTYPE

    node_data = pd.read_csv("E:\\gcn2defect\\data\\" + dataset.name + "\\Process-Binary.csv")
    edgelist = pd.read_csv(
        "E:\\gcn2defect\\data\\" + dataset.name+ "\\edges.txt", sep="\t", header=None, names=["target", "source"], dtype=nodes_dtype
    )
    node_data.apply(pd.to_numeric, errors='ignore')

    # 0 to buggy, 1 to clean
    subjects_num = node_data['bug']
    label_list = subjects_num.to_list()
    labels = []
    for i in range(len(label_list)):
        if label_list[i] == 1:
            labels.append('buggy')
        else:
            labels.append('clean')
    subjects = pd.Series(labels, dtype='str')

    cls = StellarDiGraph if directed else StellarGraph

    features = node_data.iloc[:, 3:-1]
    feature_names = node_data.iloc[:, 2]
    minMax = preprocessing.MinMaxScaler()
    features_std = minMax.fit_transform(features)

    graph = cls({"class": features_std}, {"to": edgelist})

    if edge_weights is not None:
        # A weighted graph means computing a second StellarGraph after using the unweighted one to
        # compute the weights.
        edgelist["weight"] = edge_weights(graph, subjects, edgelist)
        graph = cls({"class": node_data[feature_names]}, {"to": edgelist})

    if largest_connected_component_only:
        cc_ids = next(graph.connected_components())
        return graph.subgraph(cc_ids), subjects[cc_ids]

    return graph, subjects
```

### Run the experiment

---

After generating the CGCN embeddings, we can run the downstream task by executing pipeline.py.

### Requirements:  

------

python==3.7  
stellargraph==1.2.1  
tensorflow-gpu==2.0.1  
scikit-learn==1.0.2  
networkx==2.6.3  

### Dataset

------

| projects | version | nodes | defective rate |
| -------- | ------- | ----- | :------------- |
| ant      | 1.4     | 175   | 22.86%         |
| ant      | 1.6     | 343   | 26.82%         |
| ant      | 1.7     | 732   | 22.40%         |
| camel    | 1.2     | 578   | 36.68%         |
| camel    | 1.4     | 805   | 18.01%         |
| camel    | 1.6     | 886   | 21.22%         |
| jedit    | 3.2     | 260   | 34.62%         |
| jedit    | 4.0     | 293   | 25.60%         |
| jedit    | 4.1     | 299   | 26.42%         |
| lucene   | 2.0     | 181   | 50.28%         |
| lucene   | 2.2     | 229   | 62.45%         |
| lucene   | 2.4     | 324   | 62.35%         |
| poi      | 1.5     | 228   | 60.53%         |
| poi      | 2.5     | 371   | 65.77%         |
| poi      | 3.0     | 427   | 65.34%         |
| velocity | 1.4     | 192   | 76.04%         |
| velocity | 1.5     | 212   | 66.51%         |
| velocity | 1.6     | 227   | 34.36%         |
| xalan    | 2.4     | 676   | 16.12%         |
| xalan    | 2.5     | 725   | 50.76%         |
| xalan    | 2.6     | 810   | 46.17%         |
