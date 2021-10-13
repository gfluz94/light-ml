### LIGHT-ML

This is a library developed to incorporate useful steps in every data science and machine learning project, in order to facilitate and accelerate model development. Therefore, data scientists can spend less time working on coding preprocessing methods/scripts and use this time more wisely to create new features and tune the best model.


### Example with Customized Scikit-Learn Preprocessors

The main purpose here is to show how the objects made available by the module `light_ml.preprocessors` can be readily used in feature selection - more specifically, we will apply Boruta feature selection technique.

First let's import some usual packages and use `iris` dataset in order to show how our library can be used in this context.  

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```  

Then we can build the dataset and subsequently perform train-test split:  

```python
data = load_iris()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = pd.Series(data["target"], name="target")
```

Finally, we can import and instantiate our feature selection object:  

```python
from light_ml.preprocessor import BorutaFeatureSelector

bfs = BorutaFeatureSelector(trials=50, percentile=0.01, keep_only_tail=False)
```

The final step is then to train our transformer and use some of its methods and properties:  

```python
bfs.fit(X_train, y_train)
```

+ Summary of the feature selection procedure:  

```python
bfs.summary()
```


<pre>**************************************************
*                    SUMMARY                     *
**************************************************

&gt;&gt; Features to drop (&lt;= 17):
	* sepal length (cm)    [hits: 4]
	* sepal width (cm)     [hits: 0]

&gt;&gt; Features to tentatively keep (17 &lt; hits &lt; 33):
	

&gt;&gt; Features to drop (&gt;= 33):
	* petal length (cm)    [hits: 49]
	* petal width (cm)     [hits: 50]

</pre>


+ Selected Features:  

```python
bfs.selected_features
```


<pre>[&#39;petal length (cm)&#39;, &#39;petal width (cm)&#39;]</pre>


+ Visualization of the decision regions:  

```python
bfs.show_decision_regions(show_features=True)
```

+ Transforming our dataset:  

```python
bfs.transform(X_train)
```

<div>
    <table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            <th></th>
            <th>petal length (cm)</th>
            <th>petal width (cm)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <th>26</th>
            <td>1.6</td>
            <td>0.4</td>
            </tr>
            <tr>
            <th>8</th>
            <td>1.4</td>
            <td>0.2</td>
            </tr>
            <tr>
            <th>133</th>
            <td>5.1</td>
            <td>1.5</td>
            </tr>
            <tr>
            <th>101</th>
            <td>5.1</td>
            <td>1.9</td>
            </tr>
            <tr>
            <th>15</th>
            <td>1.5</td>
            <td>0.4</td>
            </tr>
            <tr>
            <th>...</th>
            <td>...</td>
            <td>...</td>
            </tr>
            <tr>
            <th>130</th>
            <td>6.1</td>
            <td>1.9</td>
            </tr>
            <tr>
            <th>84</th>
            <td>4.5</td>
            <td>1.5</td>
            </tr>
            <tr>
            <th>17</th>
            <td>1.4</td>
            <td>0.3</td>
            </tr>
            <tr>
            <th>56</th>
            <td>4.7</td>
            <td>1.6</td>
            </tr>
            <tr>
            <th>78</th>
            <td>4.5</td>
            <td>1.5</td>
            </tr>
        </tbody>
    </table>
</div>