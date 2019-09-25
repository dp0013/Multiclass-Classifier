# Multi-class Classifier
 This is a Neural Net model using Octave to recognize handwritten numbers, based on the Coursera MAchine Learning course.
 
 ===== DESCRIPTION ===== 
 
These are OCTAVE files (i.e., similar to MATLAB), largely based on matrix operations implementing logistic regression and neural network algorithms to predict which handwritten number is being displayed.  The hand-written digits are provided in a sample set of 5000 20 x 20 grayscale pixel images, thus the total size of the data (sample) set (X) is 5000 x 400, with each value representing a brightness for that particular pixel.
 
The program then uses forward propogation through 3 layers (one input, one hidden, and one output) to predict which number is being displayed with a 97.5% accuracy.
 
 
===== HOW TO USE =====

There are two datasets available upon which to classify:  the first is provided by Stanford University while the second consists of 6 jpg photos of random numbers I personally wrote by hand.

1)  To Use Stanford Examples:  Type (without quotes) "numberRecognition" in the Octave Command-Line Interface (CLI).  The program will generate each digit individually in a separate window (Figure 1) and will display the corresponding predicted classification (value) in the CLI.  Press 'enter' to try a new one or 'q' to quit.

2)  To Use My Examples:  Type "vectorImage = customImage('fileName.jpg', cropPercentage=0, rotStep=0)", where fileName is the name of the specific number you want to classify.  Upon completion of this step, you will see a computer-rendered, 20x20 pixel, grayscale version of the digit.  Then, type "predict(Theta1, Theta2, vectorImage)".  You will then see the classification in the command prompt (i.e., what the computer believes the number to be).  


===== CREDIT =====

This project was based on week 4 programming assignment of Coursera's Machine Learning course by Stanford University's Andrew Ng.  Enjoy! 
