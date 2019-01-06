# 深度學習資源 Deep Learning Resources
這裡紀錄了我在學習 [Deep Learning](https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0) 時蒐集的一些線上資源。內容由淺入深，希望能幫助到想深入了解這領域的你：）

## 本文內容
- [TODO] 等全部小節名稱固定後

## 遊樂場 Playground
這裡列舉一些透過瀏覽器就能開始遊玩 / 體驗 / 學習 Deep Learning 的 App 及網站，適合想要快速了解深度學習能做些什麼的人。

|[Deep Playground](https://playground.tensorflow.org/)|[ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/index.html)|
|:---:|:---:|
|<a href="https://playground.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/deep-playground.jpg"></a>|<a href="https://playground.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/convnetjs.jpg"></a>

- [Deep Playground](https://playground.tensorflow.org/)
    - 由 [Tensorflow 團隊](https://github.com/tensorflow/playground)推出，模擬訓練一個[類神經網路](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)的過程並了解其運作原理
    - 可以搭配這篇 [Introduction to Neural Networks: Playground Exercises](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/playground-exercises) 學習
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
    - 訓練類神經網路來解決經典的 [MNIST 手寫數字辨識問題](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)、[圖片生成](https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html)以及[增強式學習](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
    - 由[特斯拉汽車 Tesla](https://www.tesla.com/zh_TW/) 的 AI 負責人 [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/) 建立

|[Magenta](https://magenta.tensorflow.org/)|[Google AI Experiments](https://experiments.withgoogle.com/collection/ai)|
|:---:|:---:|
|<a href="https://magenta.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/magenta.jpg"></a>|<a href="https://experiments.withgoogle.com/collection/ai"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/google-ai-experiment.jpg"></a>

<br/>

- [Magenta](https://magenta.tensorflow.org/) 
    - 一個利用[機器學習](https://zh.wikipedia.org/zh-hant/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)來協助人們進行音樂以及藝術創作的開源專案
    - 可以在網站上的 [Demo 頁面](https://magenta.tensorflow.org/demos)嘗試各種由深度學習驅動的音樂 / 繪畫應用（如彈奏鋼琴、擊鼓）
- [Google AI Experiments](https://quickdraw.withgoogle.com/)
    - 這邊展示了接近 40 個利用圖片、語言以及音樂來與使用者產生互動的機器學習 Apps，非常值得慢慢探索
    - 知名例子有 [Quick Draw](https://quickdraw.withgoogle.com/) 以及 [Embedding Projector](https://projector.tensorflow.org/)，將在下方介紹

<br/>

|[Quick Draw](https://quickdraw.withgoogle.com/)|[Embedding Projector](https://projector.tensorflow.org/)|
|:---:|:---:|
|<a href="https://quickdraw.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/quickdraw.jpg"></a>|<a href="https://projector.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/embedding-projector.jpg"></a>

- [Quick Draw](https://teachablemachine.withgoogle.com/)
    - 由 Google 推出的知名手寫塗鴉辨識，使用的神經網路架構有常見的[卷積神經網路 CNN](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)以及[循環神經網路 RNN](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)
    - 該深度學習模型會不斷將最新的筆觸當作輸入來預測使用者想畫的物件。你會驚嘆於她精準且即時的判斷
- [Embedding Projector](https://projector.tensorflow.org/)
    - 我們時常將圖片、文字轉成[高維向量 Embedding](https://en.wikipedia.org/wiki/Tensor)以供神經網路處理，而 Projector 能將此高維向量投影到 2、3 維空間上方便我們理解
    - Projector 是 [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) 的[內建功能](https://www.tensorflow.org/guide/embedding)，你可以在這邊了解更多

<br/>


https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E8%A9%9E%E5%90%91%E9%87%8F%EF%BC%9A%E5%B0%87%E8%A9%9E%E5%BD%99%E8%A1%A8%E9%81%94%E6%88%90%E6%9C%89%E6%84%8F%E7%BE%A9%E7%9A%84%E5%90%91%E9%87%8F


## 線上教材 Courses / Tutorials

在看過上面的遊樂場以後，我相信你會有些 idea 想要實現，這時候你會需要理論 / 程式基礎

- [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
    - A hosted Jupyter notebook environment that is free to use and requires no setup. Used to demonstrate the 
- seedbank

| | |
|:-------------------------:|:-------------------------:|
|<a href="https://colab.research.google.com/notebooks/welcome.ipynb" target="_blank">Colaboratory</a> <br/><br/> ![](images/learning-resources/colab.jpg)  | <a href="https://cs.stanford.edu/people/karpathy/convnetjs/index.html" target="_blank">ConvNetJS</a> <br/><br/> ![](images/playground/convnetjs.jpg)





## Framework


## 實用工具


- Colab

## 部落格


## Newsletter


## Colab notebook

## Terminology
- leemeng.tw/dl-terminology.html


<p align="center">
  <a href="https://damp-anchorage-60936.herokuapp.com/" target="_blank"><img src="images/cover.png" alt="Cover" width="60%"/></a>
</p>