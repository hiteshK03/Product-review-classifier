# Product-review-classifier
Rate product reviews from 1 to 5

# Steps to run the program:

## Initial Setup:
<ol> 
	<li> Install virtual env if not present already using `sudo apt install python3-venv` </li>
	<li> Create a virutal environment using `python3.6 -m venv nlp772`</li>
	<li> Source the created virtualenv using `source nlp772/bin/activate` </li>
	<li> Install all the required packages using pip: `pip install -r requirements.txt` </li>
	<li> Install pytorch and torchtext with `pip install --pre torch torchtext -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html` </li>
	<li> Download spcy model using `python -m spacy download en_core_web_sm` </li>
	<li> Download embeddings from http://nlp.stanford.edu/data/glove.840B.300d.zip </li>
</ol>

## Running the code:

### NeuralNet

For training, one can run the colab file

### RNN

Running the <b>auto.py</b> file will popup a UI and user can use/change the parameters as they wish

### Transformers

Running kivy_main.py will start a GUI in which user can input sentences for review ratings  

For training, one can run the colab file or ```python sequence_classifiction.py```