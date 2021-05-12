import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from main import give_preds


class btnImage(ButtonBehavior, Image):
    def __init__(self, *args, **kwargs):
        super(btnImage, self).__init__(*args, **kwargs)


class MyGridLayout(FloatLayout):
    # Initialize infinite keywords
    def __init__(self, **kwargs):
        # Call grid layout constructor
        super(MyGridLayout, self).__init__(**kwargs)
        self.makeHome()

    def makeHome(self, *args, **kwargs):
        # Add widgets
        self.inputLabel = Label(text="Input Sentence: ", font_size=30, size_hint=(0.4, 0.08), pos_hint={"center_x": 0.25, "center_y": 0.9})
        self.add_widget(self.inputLabel)

        self.inputText = TextInput(multiline=False, font_size=30, size_hint=(0.5, 0.08), pos_hint={"center_x": 0.7, "center_y": 0.9})
        self.inputText.bind(on_text_validate=self.inputValidate)
        self.add_widget(self.inputText)

        self.evalBtn = Button(text="Evaluate", font_size=30, size_hint=(0.3, 0.08), pos_hint={"center_x": 0.25, "center_y": 0.8}, on_release=self.calcResult)
        self.add_widget(self.evalBtn)

        self.inputWeights = TextInput(multiline=False, font_size=30, size_hint=(0.5, 0.08), pos_hint={"center_x": 0.7, "center_y": 0.8})
        self.inputWeights.bind(on_text_validate=self.weightsValidate)
        self.add_widget(self.inputWeights)

        self.proLabel = Label(text="Input text goes here", font_size=30, size_hint=(0.6, 0.08), pos_hint={"center_x": 0.5, "center_y": 0.7})
        self.add_widget(self.proLabel)

        self.weightsText = Label(text="", font_size=30, size_hint=(0.5, 0.08), pos_hint={"center_x": 0.7, "center_y": 0.6})
        self.add_widget(self.weightsText)

        self.outputBox = GridLayout(cols=2, size_hint=(0.4, 0.6), pos_hint={"center_x": 0.25, "center_y": 0.35})
        self.add_widget(self.outputBox)

        self.classNameLabels = []
        self.classProbLabels = []
        self.numClasses = 5
        for i in range(self.numClasses):
            lbl = Label(text="class : " + str(i), font_size=30, size_hint=(1, 1))
            self.outputBox.add_widget(lbl)
            self.classNameLabels.append(lbl)
            lbl = Label(text="-", font_size=30, size_hint=(1, 1))
            self.outputBox.add_widget(lbl)
            self.classProbLabels.append(lbl)


        return

    def inputValidate(self, text, *args, **kwargs):
        self.proText = self.inputText.text
        self.proLabel.text = "test sentence acquired : " + self.proText

    def weightsValidate(self, text, *args, **kwargs):
        self.proWeights = self.inputWeights.text
        self.proLabel.text = "weights path acquired : " + self.proWeights

    def calcResult(self, *args, **kwargs):
        self.confMat = btnImage(source='confusion_matrix.png', size_hint=(0.5, 0.5), pos_hint={"center_x": 0.7, "center_y": 0.3})
        self.add_widget(self.confMat)
        self.proLabel.text = "evaluating sentence on selected model..."
        self.weightsText.text = "LSTM selected"
        probs = self.giveProbs(self.proText, self.proWeights)
        best_class = 0
        for i in range(self.numClasses):
            if probs[best_class] < probs[i]:
                best_class = i
            self.classProbLabels[i].text = str(probs[i])
        self.proLabel.text = "sentence belongs to class : " + str(best_class)

    def giveProbs(self, text, weights, *args, **kwargs):
        self.proLabel.text = "evaluated sentence"
        pred_arr = give_preds(text, weights)
        return pred_arr


class MyApp(App):
    def build(self):
        return MyGridLayout()


if __name__ == '__main__':
    MyApp().run()
