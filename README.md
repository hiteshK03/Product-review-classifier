# Product-review-classifier-CS772-DL4NLP
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

### Transformers with Lime Explainability

Running kivy_main.py will start a GUI in which user can input sentences for review ratings  

For training, one can run the colab file or ```python sequence_classifiction.py```

### Lime example
![lime_analysis_plot](https://user-images.githubusercontent.com/45922320/118012760-06b1f600-b36f-11eb-9c01-ded82c49b07c.png)

