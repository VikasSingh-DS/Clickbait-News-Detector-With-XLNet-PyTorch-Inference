# Clickbait-News-Detector-With-XLNet-PyTorch

## Introdution
Online content publishers often use catchy headlines for their articles in order to attract users to their websites. These headlines, popularly known as clickbaits, exploit a user’s curiosity gap and lure them to click on links that often disappoint them. Existing methods for automatically detecting clickbaits rely on heavy feature engineering and domain knowledge. Here, my goal is to classification of clickbait news and non-clicbait news. I have fine-tune SOTA XLNet model for classification. I have used the amazing Transformers library by Hugging Face with pytorch.

## Dataset
The train1.csv collected from Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly. "Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media”. In Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), San Fransisco, US, August 2016.GitHub

It has two columns first one contains headlines and the second one has numerical labels of clickbait in which 1 represents that it is clickbait and 0 represents that it is non-clickbait headline. The dataset contains total 32000 rows of which 50% are clickbait and other 50% are non-clickbait.

The train2.csv collected from Clickbait news detection dataset from the Kaggle InClass Prediction Competition. The dataset contains title and text of the news and label.

## XLNet Model
XLNet is the latest and greatest model to emerge from the booming field of Natural Language Processing (NLP). The XLNet paper combines recent advances in NLP with innovative choices in how the language modelling problem is approached. When trained on a very large NLP corpus, the model achieves state-of-the-art performance for the standard NLP tasks that comprise the GLUE benchmark. To learn more about the model see the link.

## XLNet Fine-Tuning With PyTorch
I'll use XLNet with the huggingface PyTorch library to quickly and efficiently fine-tune a model to get near state of the art performance in sentence classification. More broadly, I describe the practical application of transfer learning in NLP to create high performance models with minimal effort on a range of NLP tasks. Specifically, we will take the pre-trained BERT model, add an untrained layer of neurons on the end, and train the new model for our classification task.

## Reference
[zihangdai/xlnet](https://github.com/zihangdai/xlnet)
[Hugging Face](https://huggingface.co/transformers/model_doc/bert.html),
[papers with code](https://paperswithcode.com/task/clickbait-detection),
[Training Sentiment Model Using BERT-By Abhishek Thakur](https://www.youtube.com/watch?v=hinZO--TEk4&t=449s),
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/),
[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/),
[How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf),
[BERT Explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
