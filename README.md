##### IFT799 - TP 3 & 4
##### Gabriel Gibeau Sanchez - gibg2501
##### 2020-12-11


### install packages
All needed packages can be installed using:

```python
pip install -r requirements.txt
```

### Models
Model 1 and 4 are implemented in tp3.py. Model 3 is implemented in tp3_model3.py

### Serialized object
In order to avoid redundant calculations, some objects are serialized using Pickle. The python script looks for 
pickle object in the working directory, and loads them if they are available. If not, it performs the elbow method 
and some euclidian distance calculation from scratch. The pickle object stored in ./pickle_objects can be copied to the root directory for reuse.
