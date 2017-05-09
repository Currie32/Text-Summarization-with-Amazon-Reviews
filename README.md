# Text-Summarization-with-Amazon-Reviews

The objective of this project is to build a seq2seq model that can create relevant summaries for reviews written about fine foods sold on Amazon. This dataset contains above 500,000 reviews, and is hosted on [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews). It's too large to host here, it's over 300MB.

To build our model we will use a two-layered bidirectional RNN with LSTMs on the input data and two layers, each with an LSTM using bahdanau attention on the target data. [Jaemin Cho's tutorial](https://github.com/j-min/tf_tutorial_plus/tree/master/RNN_seq2seq/contrib_seq2seq) for seq2seq was really helpful to get the code in working order because this is my first project with TensorFlow 1.1; some of the functions are very different from 1.0. The architecture for this model is similar to Xin Pan's and Peter Liu's, here's their [GitHub page.](https://github.com/tensorflow/models/tree/master/textsum)

This model uses [Conceptnet Numberbatch's](https://github.com/commonsense/conceptnet-numberbatch) pre-trained word vectors. 

Here are some examples of reviews and their generated summaries:
- Description(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
- Summary(1): great coffee

- Description(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won’t either!
- Summary(2): omg gross gross

- Description(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
- Summary(3): love it
