<div align="center">
  <img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/general/paper-ball.jpg">
</div>

-----------------
這裡紀錄了我在學習[深度學習](https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)時蒐集的一些線上資源。內容由淺入深，而且會不斷更新，希望能幫助你順利地開始學習：）

## 本文章節
- [遊玩空間](#playground)
- [線上課程](#courses)
- [實用工具](#tools)
- [其他教材](#tutorials)
- [優質文章](#blogs)
- [經典論文](#papers)

## <div id='playground'>遊玩空間</div>
這節列舉了一些透過瀏覽器就能馬上開始遊玩 / 體驗深度學習的應用。作為這些應用的使用者，你可以先高層次、直觀地了解深度學習能做些什麼。之後有興趣再進一步了解背後原理。

這小節最適合：

- 想要快速體會深度學習如何被應用在真實世界的好奇寶寶
- 想要直觀理解[類神經網路（Artifical Neural Network）](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)運作方式的人
- 想從別人的深度學習應用取得一些靈感的開發者

|[Deep Playground](https://playground.tensorflow.org/)|[ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/index.html)|
|:---:|:---:|
|<a href="https://playground.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/deep-playground.jpg"></a>|<a href="https://cs.stanford.edu/people/karpathy/convnetjs/index.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/convnetjs.jpg"></a>

### [Deep Playground](https://playground.tensorflow.org/)

- 由 [Tensorflow 團隊](https://github.com/tensorflow/playground)推出，模擬訓練一個類神經網路的過程並了解其運作原理
- 可以搭配這篇 [Introduction to Neural Networks: Playground Exercises](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/playground-exercises) 學習

### [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)

- 訓練類神經網路來解決經典的 [MNIST 手寫數字辨識問題](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)、[圖片生成](https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html)以及[增強式學習](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- 由 Tesla 的 AI 負責人 [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/) 建立

|[Magenta](https://magenta.tensorflow.org/)|[Google AI Experiments](https://experiments.withgoogle.com/collection/ai)|
|:---:|:---:|
|<a href="https://magenta.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/magenta.jpg"></a>|<a href="https://experiments.withgoogle.com/collection/ai"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/google-ai-experiment.jpg"></a>

### [Magenta](https://magenta.tensorflow.org/) 

- 一個利用[機器學習](https://zh.wikipedia.org/zh-hant/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)來協助人們進行音樂以及藝術創作的開源專案
- 可以在網站上的 [Demo 頁面](https://magenta.tensorflow.org/demos)嘗試各種由深度學習驅動的音樂 / 繪畫應用（如彈奏鋼琴、擊鼓）

### [Google AI Experiments](https://experiments.withgoogle.com/collection/ai)

- 這邊展示了接近 40 個利用圖片、語言以及音樂來與使用者產生互動的機器學習 Apps，值得慢慢探索
- 知名例子有 [Quick Draw](https://quickdraw.withgoogle.com/) 以及 [Teachable Machine](https://teachablemachine.withgoogle.com/)，將在下方介紹

|[Quick Draw](https://quickdraw.withgoogle.com/)|[Teachable Machine](https://teachablemachine.withgoogle.com/)|
|:---:|:---:|
|<a href="https://quickdraw.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/quickdraw.jpg"></a>|<a href="https://teachablemachine.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/teachable-machine.jpg"></a>

### [Quick Draw](https://quickdraw.withgoogle.com/)

- 由 Google 推出的知名手寫塗鴉辨識，使用的神經網路架構有常見的[卷積神經網路 CNN ](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)以及[循環神經網路 RNN](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)
- 該深度學習模型會不斷將最新的筆觸當作輸入來預測使用者想畫的物件。你會驚嘆於她精準且即時的判斷

### [Teachable Machine](https://teachablemachine.withgoogle.com/)
- 利用電腦 / 手機上的相機來訓練能將影像對應到其他圖片、音訊的神經網路，饒富趣味
- 透過這例子，你將暸解機器學習的神奇之處以及其侷限所在

|[Fast Neural Style](https://tenso.rs/demos/fast-neural-style/)| [TensorFlow.js](https://js.tensorflow.org/)
|:---:|:---:|
|<a href="https://tenso.rs/demos/fast-neural-style/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/fast-neural-style.jpg"></a>|<a href="https://js.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/human-pose-estimation.jpg"></a>

### [Fast Neural Style](https://tenso.rs/demos/fast-neural-style/)

- 展示如何使用 WebGL 在瀏覽器快速地進行[神經風格轉換 Neural Style Transfer](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- 你可以選擇任何一張圖片，並在此網站上將其畫風轉變成指定的藝術照
- [Deepart.io](https://deepart.io/) 也提供類似服務

### [TensorFlow.js](https://js.tensorflow.org/)

- TensorFlow.js 頁面有多個利用 JavaScript 實現的深度學習應用，如上圖中的[人類姿勢估計 Human Pose Estimation](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)。
- 你可以在該應用裡頭打開自己的攝影機，看該應用能不能偵測到你與朋友的姿勢。

## <div id='courses'>線上課程</div>
看完[遊玩空間](#playground)的大量實際應用，相信你已經迫不及待地想要開始學習強大的深度學習技術了。

這節列舉了一些有用的線上課程以及學習教材，幫助你掌握深度學習的基本知識（沒有特別註明的話皆為免費存取）。

另外值得一提的是，大部分課程都要求一定程度的 [Python](https://www.python.org/) 程式能力。

|[李宏毅教授的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)| [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)
|:---:|:---:|
|<a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/Hung-Yi-Lee-ml-courses.jpg"></a>|<a href="https://www.coursera.org/specializations/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/deep-learning-specification-coursera.jpg"></a>

### [李宏毅教授的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

- 大概是全世界最好、最完整的 Deep Learning <b>中文</b>學習資源。
- 影片內容涵蓋基本理論（約 10 小時觀看時間）一直到進階的[生成對抗網路 GAN](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C) 以及[強化學習 RL](https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)
- 你也可以從[這邊](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)看到教授的 Youtube 課程清單

### [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)

- 原 Google Brain 的[吳恩達](https://zh.wikipedia.org/wiki/%E5%90%B4%E6%81%A9%E8%BE%BE)教授開授的整個深度學習專項課程共分五堂課，從[神經網路的基礎](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)到能夠進行機器翻譯、語音辨識的[序列模型](https://www.coursera.org/learn/nlp-sequence-models)，每堂課預計 1 個月完成，收費採訂閱制
- 程式作業會交互使用 [Numpy](http://www.numpy.org/)、[Keras](https://keras.io/) 以及 [TensorFlow](https://www.tensorflow.org/) 來實作深度學習模型

|[Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)| [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)
|:---:|:---:|
|<a href="https://course.fast.ai/index.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/fast-ai.jpg"></a>|<a href="https://www.kaggle.com/learn/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/kaggle-learn-dl.jpg"></a>

### [Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)

- 7 週課程，一週約需安排 10 小時上課。該課程由[傑里米·霍華德](https://www.kaggle.com/jhoward)來講解深度學習，其在知名數據建模和數據分析競賽平台 [Kaggle](https://www.kaggle.com/) 維持兩年的世界第一

### [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)

- 14 堂課程，主要使用 TensorFlow 實作深度學習模型
- 內容主要專注在[電腦視覺（Computer Vision）](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)以及如何應用[遷移學習（Transfer Learning）](https://en.wikipedia.org/wiki/Transfer_learning)

|[Elements of Artificial Intelligence](https://www.elementsofai.com/)| To Be Updated
|:---:|:---:|
|<a href="https://www.elementsofai.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/elementsofai.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/general/to-be-updated.jpg"></a>

### [Elements of Artificial Intelligence](https://www.elementsofai.com/)

- 芬蘭最高學府[赫爾辛基大學](https://zh.wikipedia.org/wiki/%E8%B5%AB%E5%B0%94%E8%BE%9B%E5%9F%BA%E5%A4%A7%E5%AD%A6)推出的 AI 課程。此課程目的在於讓所有人都能了解 AI，不需要任何程式經驗。這堂課非常適合完全沒有接觸過深度學習或是相關領域的人
- 課程分 6 個部分，包含「何謂 AI ？」、「真實世界的 AI」、「機器學習」以及「神經網路」等章節

## <div id="tools">實用工具</div>
這節列出一些在你的深度學習路上可以幫得上些忙的工具。

|[Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)| [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
|:---:|:---:|
|<a href="https://colab.research.google.com/notebooks/welcome.ipynb"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/colab.jpg"></a>|<a href="https://www.tensorflow.org/guide/summaries_and_tensorboard"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/tensorboard.jpg"></a>

### [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)

- 由 Google 提供的雲端 [Jupyter](https://jupyter.org/) 筆記本環境，讓你只要用瀏覽器就能馬上開始訓練深度學習模型。你甚至還可以使用一個免費的 [Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/) GPU 或 [TPU](https://colab.research.google.com/notebooks/tpu.ipynb) 來加速訓練自己的模型
- 該計算環境也能與自己的 [Google Drive](https://colab.research.google.com/notebooks/io.ipynb) 做連結，讓運算雲端化的同時將筆記本 / 模型結果都同步到自己的筆電上

### [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

- TensorBoard 是一個視覺化工具，方便我們了解、除錯並最佳化自己訓練的深度學習模型
- 除了 TensorFlow 以外，其他基於 Python 的機器學習框架大多也可以透過 [tensorboardX](https://github.com/lanpa/tensorboardX) 來使用 TensorBoard


|[Embedding Projector](https://projector.tensorflow.org/)|[Lucid](https://github.com/tensorflow/lucid)|
|:---:|:---:|
|<a href="https://projector.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/embedding-projector.jpg"></a>|<a href="https://github.com/tensorflow/lucid"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/lucid.jpg"></a>|

### [Embedding Projector](https://projector.tensorflow.org/)

- 我們時常需要將圖片、文字轉成[高維數字向量 Embedding](https://en.wikipedia.org/wiki/Tensor) 以供神經網路處理，而 Projector 能將此高維向量投影到 2、3 維空間上方便我們理解這些數據
- Projector 網站讓你在線上探索幾個常見的資料集，但事實上你也可以[利用 Tensorboard 來視覺化自己的數據](https://www.tensorflow.org/guide/embedding)。

### [Lucid](https://github.com/tensorflow/lucid)

- Lucid 是一個嘗試讓神經網路變得更容易解釋的開源專案，裡頭包含了很多視覺化神經網路的筆記本
- 你可以直接在 Colab 上執行這些筆記本並了解如何視覺化神經網路

## <div id="tutorials">其他教材</div>
除了[線上課程](#courses)以外，網路上還有無數的學習資源。

這邊列出一些推薦的深度學習教材，大多數皆以數據科學家常用的 [Jupyter](https://jupyter.org/) 筆記本的方式呈現。

你可以將感興趣的筆記本導入[實用工具](#tools)裡提到的 [Colaboratory（Colab）](https://colab.research.google.com/notebooks/welcome.ipynb)，馬上開始學習。

|[Seedbank](https://research.google.com/seedbank/)| [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)
|:---:|:---:|
|<a href="https://research.google.com/seedbank/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/seedbank.jpg"></a>|<a href="https://github.com/fchollet/deep-learning-with-python-notebooks"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/fchollet-deep-learning-with-python.jpg"></a>

### [Seedbank](https://research.google.com/seedbank/)

- 讓你可以一覽 Colab 上超過 100 個跟機器學習相關的筆記本，並以此為基礎建立各種深度學習應用
- 熱門筆記本包含[神經機器翻譯](https://research.google.com/seedbank/seed/5695159920492544)、[音樂生成](https://research.google.com/seedbank/seed/5681034041491456)以及 [DeepDream](https://research.google.com/seedbank/seed/5631986051842048)
- 因為是 Google 服務，筆記本大多使用 TensorFlow 與 Keras 來實現模型
    
### [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)

- [Keras](https://keras.io/) 作者 [François Chollet](https://ai.google/research/people/105096) 在 [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438) 一書中用到的所有筆記本。每個筆記本裡頭都清楚地介紹該如何使用 Keras 來實現各種深度學習模型，十分適合第一次使用 Python 實現深度學習的讀者 
- [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#top)一文的 Keras 程式碼大多基於此
- 附註：Keras 在 TensorFlow 2.0 中[將成為其最重要的高層次 API](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)

|[Stanford CS230 Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)|[practicalAI](https://github.com/GokuMohandas/practicalAI)|
|:---:|:---:|
|<a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/cs230-deep-learning-cheatsheet.jpg"></a>|<a href="https://github.com/GokuMohandas/practicalAI"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/practical-ai-pytorch.jpg"></a>

### [Stanford CS230 Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

- 史丹佛大學的[深度學習課程 CS230](http://cs230.stanford.edu/) 釋出的深度學習小抄總結了目前最新的[卷積神經網路](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)及[循環神經網路](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)知識，還包含了[訓練深度學習時需要使用到的技巧](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)，十分強大
- 此小抄最適合已經熟悉基礎知識的同學隨時複習運用。你也可以從他們的 [Github Repo](https://github.com/afshinea/stanford-cs-230-deep-learning) 下載包含上述所有內容的[超級 VIP 小抄](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/master/en/super-cheatsheet-deep-learning.pdf)
- 除了深度學習以外，你也可以查看 [CS229 機器學習課程的小抄](https://stanford.edu/~shervine/teaching/cs-229.html)

### [practicalAI](https://github.com/GokuMohandas/practicalAI)
- 在 Github 上超過 1 萬星的 Repo。除了深度學習，也有介紹 [Python 基礎](https://colab.research.google.com/github/GokuMohandas/practicalAI/blob/master/notebooks/01_Python.ipynb)及 [Pandas](https://colab.research.google.com/github/GokuMohandas/practicalAI/blob/master/notebooks/03_Pandas.ipynb) 的使用方式
- 使用 [Pytorch](https://pytorch.org/) 框架來實現深度學習模型，且所有內容都是 Jupyter 筆記本，可以讓你在 Colab 或本地端執行

|[AllenNLP Demo](http://demo.allennlp.org/)| To Be Updated
|:---:|:---:|
|<a href="http://demo.allennlp.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/allennlp-demo.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/general/to-be-updated.jpg"></a>

### [AllenNLP Demo](http://demo.allennlp.org/)

- 清楚地展示了如[機器理解](https://demo.allennlp.org/machine-comprehension)、[命名實體識別](https://demo.allennlp.org/named-entity-recognition)等多個自然語言處理任務的情境。每個任務的情境包含了任務所需要的輸入、SOTA 模型的預測結果以及模型內部的注意力機制，對理解一個 NLP 任務的實際應用情境有很大幫助
- [AllenNLP](https://allennlp.org/) 是一個由 [AI2](https://allenai.org/) 以 [PyTorch](https://pytorch.org/) 實現的自然語言處理函式庫

## <div id="blogs">優質文章</div>
這邊列舉了一些幫助我釐清重要概念的部落格以及網站，希望能加速你探索這個深度學習世界。

只要 Google 一下就能發現這些部落格裡頭很多文章都有中文翻譯。但為了尊重原作者，在這邊都列出原文連結。

- [Distill](https://distill.pub/about/)
    - 用非常高水準且互動的方式來說明複雜的深度學習概念。[Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)、[Ian Goodfellow](http://www.iangoodfellow.com/) 及 [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) 等知名人士皆參與其中
- [R2D3: 圖解機器學習](http://www.r2d3.us/%E5%9C%96%E8%A7%A3%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AC%AC%E4%B8%80%E7%AB%A0/)
    - 利用非常直覺易懂的視覺化來說明機器學習，連結為中文版
- [Christopher Olah's blog](http://colah.github.io/)
    - 詳細解釋不少深度學習概念。作者在[這篇](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)就詳細地解釋了[長短期記憶 LSTM](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E8%A8%98%E6%86%B6%E5%8A%9B%E5%A5%BD%E7%9A%84-LSTM-%E7%B4%B0%E8%83%9E) 的概念與變形；在[這篇](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)則解釋何為 CNN 的卷積運算
- [Jay Alammar's blog](https://jalammar.github.io/)
    - 以清楚易懂的視覺化解釋深度學習概念。[這篇](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)用大量易懂的動畫說明[神經機器翻譯](https://en.wikipedia.org/wiki/Neural_machine_translation)，而在[這篇](https://jalammar.github.io/illustrated-bert/)則介紹如何利用如 [ELMo](https://allennlp.org/elmo)、[BERT](https://github.com/google-research/bert) 等預先訓練過的強大模型在自然語言處理進行[遷移學習](https://en.wikipedia.org/wiki/Transfer_learning)
- [Andrej Karpathy's blog](http://karpathy.github.io/)
    - 現為 Tesla AI 負責人的 Andrej Karpathy 在[這篇](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)明確說明何謂循環神經網路 RNN。文中提供不少應用實例及視覺化來幫助我們理解 RNN 模型究竟學到了什麼，是學習 RNN 的朋友幾乎一定會碰到的一篇文章

## <div id="papers">經典論文</div>
這邊依發表時間列出深度學習領域的經典 / 重要論文。

為了幫助你快速掌握論文內容以及歷年的研究趨勢，每篇論文下會有非常簡短的介紹（WIP）。

但我們推薦有興趣的人自行閱讀論文以深入了解。

### 自然語言處理 Natural Language Processing (NLP)
- [2003/02 A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [2013/01 Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [2013/08 Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
- [2014/09 Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [2015/08 Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [2015/12 Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432)
    - 推出一套無監督式的預訓練方法。使用無標籤數據訓練後的 RNN 模型在之後的監督式任務表現更好
- [2017/06 Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - Google 推出新的神經網路架構 [Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)。這個基於自注意力機制的架構特別適合語言理解任務
- [2017/06 One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
- [2018/01 Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- [2018/02 Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - [ELMo 詞向量](https://allennlp.org/elmo)，利用兩獨立訓練的 LSTM 獲取雙向訊息
- [2018/06 Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - [OpenAI](https://blog.openai.com/language-unsupervised/) 利用無監督式預訓練以及 Transformer 架構訓練出來的模型表現在多個 NLP 任務表現良好。約使用 8 億詞彙量的資料集
- [2018/10 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    - Google 暴力美學。利用深層 Transformer 架構、2 個精心設計的預訓練任務以及約 33 億詞彙量的資料集訓練後，得到表現卓越的語言代表模型，打破 11 項 NLP 任務紀錄

### 電腦視覺 Computer Vision (CV) (In progress and to be tidied up)
- [1998/01 Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [2009/06 ImageNet: A Large-Scale Hierarchical Image Database (ImageNet)](http://www.image-net.org/papers/imagenet_cvpr09.pdf)
- [2012/12 ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [2013/11 Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN)](https://arxiv.org/abs/1311.2524)
- [2013/12 OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks (OverFeat)](https://arxiv.org/abs/1312.6229)
- [2014/06 Generative Adversarial Networks (GAN)](https://arxiv.org/abs/1406.2661)
- [2014/06 DeepFace: Closing the Gap to Human-Level Performance in Face Verification (DeepFace)](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- [2014/09 Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [2014/09 Goint deeper with convolutions (GoogLeNet)](https://arxiv.org/abs/1409.4842)
- [2014/11 Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [2015/04 Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [2015/05 U-Net: Convolutional Networks for Biomedical Image Segmentation (U-Net)](https://arxiv.org/abs/1505.04597)
- [2015/06 Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Faster R-CNN)](https://arxiv.org/abs/1506.01497)
- [2015/06 You Only Look Once: Unified, Real-Time Object Detection (YOLO)](https://arxiv.org/abs/1506.02640)
- [2015/13 Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/abs/1511.06434)
- [2015/12 SSD: Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325)
- [2015/12 Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [2016/12 YOLO9000: Better, Faster, Stronger (YOLOv2)](https://arxiv.org/abs/1612.08242)
- [2017/01 Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875)
- [2017/03 Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [2017/03 Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/abs/1703.10593)
- [2017/04 MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (MobileNets)](https://arxiv.org/abs/1704.04861)
- [2017/07 ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (ShuffleNet)](https://arxiv.org/abs/1707.01083)
- [2018/04 YOLOv3: An Incremental Improvement (YOLOv3)](https://arxiv.org/abs/1804.02767)

## 待辦事項
還有不少內容沒有整理完成，以下是作者正在整理並已經打算追加的項目：

- 深度學習術語對照表
- 值得追蹤的業界 / 學界影響人物清單
- 無圖的資源列表版本
- （你的推薦與建議）

也歡迎你推薦更多優質的學習資源以及其他寶貴意見來幫助更多人學習：）
