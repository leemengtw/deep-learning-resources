# 由淺入深的深度學習資源
這裡紀錄了我在學習[深度學習](https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)時蒐集的一些線上資源。內容由淺入深，希望能幫助到想深入了解這領域的你：）

## 本文章節
- [遊玩空間](#playground)
- [線上課程](#courses)
- [實用工具](#tools)
- [其他教材](#tutorials)


## <div id='playground'>遊玩空間</div>
這節列舉了一些透過瀏覽器就能馬上開始遊玩 / 體驗深度學習的應用。作為這些應用的使用者，你可以先高層次、直觀地了解深度學習能做些什麼。之後有興趣再進一步了解背後原理。

這小節最適合：

- 想要快速體會深度學習如何被應用在真實世界的人
- 想利用深度學習技術來創造新東西、玩遊戲的人
- 想要直觀理解[類神經網路（Artifical Neural Network）](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)運作方式的人

|[Deep Playground](https://playground.tensorflow.org/)|[ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/index.html)|
|:---:|:---:|
|<a href="https://playground.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/deep-playground.jpg"></a>|<a href="https://cs.stanford.edu/people/karpathy/convnetjs/index.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/convnetjs.jpg"></a>

- [Deep Playground](https://playground.tensorflow.org/)
    - 由 [Tensorflow 團隊](https://github.com/tensorflow/playground)推出，模擬訓練一個類神經網路的過程並了解其運作原理
    - 可以搭配這篇 [Introduction to Neural Networks: Playground Exercises](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/playground-exercises) 學習
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
    - 訓練類神經網路來解決經典的 [MNIST 手寫數字辨識問題](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)、[圖片生成](https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html)以及[增強式學習](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
    - 由 Tesla 的 AI 負責人 [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/) 建立

|[Magenta](https://magenta.tensorflow.org/)|[Google AI Experiments](https://experiments.withgoogle.com/collection/ai)|
|:---:|:---:|
|<a href="https://magenta.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/magenta.jpg"></a>|<a href="https://experiments.withgoogle.com/collection/ai"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/google-ai-experiment.jpg"></a>

- [Magenta](https://magenta.tensorflow.org/) 
    - 一個利用[機器學習](https://zh.wikipedia.org/zh-hant/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)來協助人們進行音樂以及藝術創作的開源專案
    - 可以在網站上的 [Demo 頁面](https://magenta.tensorflow.org/demos)嘗試各種由深度學習驅動的音樂 / 繪畫應用（如彈奏鋼琴、擊鼓）
- [Google AI Experiments](https://experiments.withgoogle.com/collection/ai)
    - 這邊展示了接近 40 個利用圖片、語言以及音樂來與使用者產生互動的機器學習 Apps，值得慢慢探索
    - 知名例子有 [Quick Draw](https://quickdraw.withgoogle.com/) 以及 [Teachable Machine](https://teachablemachine.withgoogle.com/)，將在下方介紹

|[Quick Draw](https://quickdraw.withgoogle.com/)|[Teachable Machine](https://teachablemachine.withgoogle.com/)|
|:---:|:---:|
|<a href="https://quickdraw.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/quickdraw.jpg"></a>|<a href="https://teachablemachine.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/teachable-machine.jpg"></a>

- [Quick Draw](https://teachablemachine.withgoogle.com/)
    - 由 Google 推出的知名手寫塗鴉辨識，使用的神經網路架構有常見的[卷積神經網路 CNN ](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)以及[循環神經網路 RNN](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)
    - 該深度學習模型會不斷將最新的筆觸當作輸入來預測使用者想畫的物件。你會驚嘆於她精準且即時的判斷
- [Teachable Machine](https://teachablemachine.withgoogle.com/)
    - 利用電腦 / 手機上的相機來訓練能將影像對應到其他圖片、音訊的神經網路，饒富趣味

|[Fast Neural Style](https://tenso.rs/demos/fast-neural-style/)| [TensorFlow.js](https://js.tensorflow.org/)
|:---:|:---:|
|<a href="https://tenso.rs/demos/fast-neural-style/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/fast-neural-style.jpg"></a>|<a href="https://js.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/human-pose-estimation.jpg"></a>

- [Fast Neural Style](https://tenso.rs/demos/fast-neural-style/)
    - 展示如何使用 WebGL 在瀏覽器快速地進行[神經風格轉換 Neural Style Transfer](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
    - 你可以在此網站選擇一張原始圖片，並將其畫風轉變成藝術照
- [TensorFlow.js](https://js.tensorflow.org/)
    - TensorFlow.js 頁面有多個利用 JavaScript 實現的深度學習應用，如上圖中的[人類姿勢估計 Human Pose Estimation](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)
    
## <div id='courses'>線上課程</div>
看完[遊玩空間](#playgrounds)的大量實際應用，相信你已經迫不及待地想要開始學習強大的深度學習技術了。

這節列舉了一些有用的線上課程以及學習教材，幫助你掌握深度學習的基本知識。另外值得一提的是，大部分課程要求一定程度的 [Python](https://www.python.org/) 程式能力。

|[李宏毅的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)| [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)
|:---:|:---:|
|<a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/Hung-Yi-Lee-ml-courses.jpg"></a>|<a href="https://www.coursera.org/specializations/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/deep-learning-specification-coursera.jpg"></a>

- [李宏毅的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)
    - 大概是全世界最好、最完整的 Deep Learning <b>中文</b>學習資源。
    - 影片內容涵蓋基本理論（約 10 小時觀看時間）一直到進階的[生成對抗網路 GAN](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C)以及[強化學習 RL](https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)
    - 你也可以從[這邊看到教授所有的 Youtube 課程播放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
- [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)
    - 原 Google Brain 的[吳恩達教授](https://zh.wikipedia.org/wiki/%E5%90%B4%E6%81%A9%E8%BE%BE)的整個專項共分五堂課，從[神經網路的基礎](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)到能夠進行機器翻譯、語音辨識的[序列模型](https://www.coursera.org/learn/nlp-sequence-models)，每堂課預計 1 個月完成，收費採訂閱制
    - 程式作業交互使用 [Numpy](http://www.numpy.org/)、[Keras](https://keras.io/) 以及 [TensorFlow](https://www.tensorflow.org/) 來實作深度學習

|[Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)| [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)
|:---:|:---:|
|<a href="https://course.fast.ai/index.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/fast-ai.jpg"></a>|<a href="https://www.kaggle.com/learn/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/kaggle-learn-dl.jpg"></a>
- [Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)
    - 7 週課程，一週約需安排 10 小時上課。由 Kaggle 競賽兩年第一的[傑里米·霍華德](https://www.kaggle.com/jhoward)講解深度學習
- [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)
    - 14 堂課程，主要使用 TensorFlow 進行實作

## <div id="tools">實用工具</div>
這節列出一些在你的深度學習路上可以幫得上些忙的工具。

|[Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)| [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
|:---:|:---:|
|<a href="https://colab.research.google.com/notebooks/welcome.ipynb"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/colab.jpg"></a>|<a href="https://www.tensorflow.org/guide/summaries_and_tensorboard"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/tensorboard.jpg"></a>

- [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
    - 由 Google 提供的雲端 [Jupyter](https://jupyter.org/) 筆記本環境，讓你只要用瀏覽器就能馬上開始訓練深度學習模型。你甚至還可以使用一個免費的 [Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/) GPU 或 [TPU](https://colab.research.google.com/notebooks/tpu.ipynb) 來加速訓練自己的模型
    - 該計算環境也能與自己的 [Google Drive](https://colab.research.google.com/notebooks/io.ipynb) 做連結，做到完全雲端化
- [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    - TensorBoard 是一個視覺化工具，方便我們了解、除錯以及最佳化自己訓練的深度學習模型
    - 除了 TensorFlow 以外，其他深度學習框架大都也有支援 TensorBoard

||[Embedding Projector](https://projector.tensorflow.org/)|
|:---:|:---:|
||<a href="https://projector.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/embedding-projector.jpg"></a>


- [Embedding Projector](https://projector.tensorflow.org/)
    - 我們時常需要將圖片、文字轉成[高維向量 Embedding](https://en.wikipedia.org/wiki/Tensor)以供神經網路處理，而 Projector 能將此高維向量投影到 2、3 維空間上方便我們理解這些數據
    - 這邊的 Projector 為一線上系統，但事實上你也可以[使用 Tensorboard 來視覺化自己的數據](https://www.tensorflow.org/guide/embedding)。





## <div id="tutorials">其他教材</div>
這邊列出各種開源的深度學習教材，大多數皆以數據科學家常用的 [Jupyter 筆記本](https://jupyter.org/)呈現。


|[Seedbank](https://research.google.com/seedbank/)| [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)
|:---:|:---:|
|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/seedbank.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/"></a>

- Seedbank
    - 提供超過 100 個筆記本
- [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)
    - 
    - 實作使用 [Keras](https://keras.io/)，在深度學習框架 [TensorFlow 2.0 中將成為其最重要的高層次 API](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)
