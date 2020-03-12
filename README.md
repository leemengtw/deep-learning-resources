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
- [其他整理](#collections)

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

---

|[Magenta](https://magenta.tensorflow.org/)|[Google AI Experiments](https://experiments.withgoogle.com/collection/ai)|
|:---:|:---:|
|<a href="https://magenta.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/magenta.jpg"></a>|<a href="https://experiments.withgoogle.com/collection/ai"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/google-ai-experiment.jpg"></a>

### [Magenta](https://magenta.tensorflow.org/) 

- 一個利用[機器學習](https://zh.wikipedia.org/zh-hant/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)來協助人們進行音樂以及藝術創作的開源專案
- 可以在網站上的 [Demo 頁面](https://magenta.tensorflow.org/demos)嘗試各種由深度學習驅動的音樂 / 繪畫應用（如彈奏鋼琴、擊鼓）

### [Google AI Experiments](https://experiments.withgoogle.com/collection/ai)

- 這邊展示了接近 40 個利用圖片、語言以及音樂來與使用者產生互動的機器學習 Apps，值得慢慢探索
- 知名例子有 [Quick Draw](https://quickdraw.withgoogle.com/) 以及 [Teachable Machine](https://teachablemachine.withgoogle.com/)，將在下方介紹

---

|[Quick Draw](https://quickdraw.withgoogle.com/)|[Teachable Machine](https://teachablemachine.withgoogle.com/)|
|:---:|:---:|
|<a href="https://quickdraw.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/quickdraw.jpg"></a>|<a href="https://teachablemachine.withgoogle.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/teachable-machine.jpg"></a>

### [Quick Draw](https://quickdraw.withgoogle.com/)

- 由 Google 推出的知名手寫塗鴉辨識，使用的神經網路架構有常見的[卷積神經網路 CNN ](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)以及[循環神經網路 RNN](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)
- 該深度學習模型會不斷將最新的筆觸當作輸入來預測使用者想畫的物件。你會驚嘆於她精準且即時的判斷

### [Teachable Machine](https://teachablemachine.withgoogle.com/)
- 利用電腦 / 手機上的相機來訓練能將影像對應到其他圖片、音訊的神經網路，饒富趣味
- 透過這例子，你將暸解機器學習的神奇之處以及其侷限所在

---

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

---

|[GAN Lab](https://poloclub.github.io/ganlab/)| [Talk to Transformer](https://talktotransformer.com/)
|:---:|:---:|
|<a href="https://poloclub.github.io/ganlab/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/gan-lab.jpg"></a>|<a href="https://talktotransformer.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/talk_to_transformer.jpg"></a>

### [GAN Lab](https://poloclub.github.io/ganlab/)

- [對抗生成網路（**G**enerative **A**dversarial **N**etwork，簡稱GAN）](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C)是非監督式學習的一種方法，通過讓兩個神經網路相互博弈的方式進行學習。此網站以 [TensorFlow.js](https://js.tensorflow.org/) 實作 GAN 中兩個神經網路的學習過程，幫助有興趣的你更直觀地理解神奇的 GAN 的運作方式

### [Talk to Transformer](https://talktotransformer.com/)

- 展示了一個由 OpenAI 推出，名為 [GPT-2 的無監督式語言模型](https://openai.com/blog/better-language-models/)。該模型以 Google 發表的神經網路架構 [Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) 為基底，在給定一段魔戒或是復仇者聯盟的文字內容，該模型可以自己生成唯妙唯俏的延伸劇情。你也可以嘗試 [AllenAI GPT-2 Explorer](https://gpt2.apps.allenai.org/?text=Joel%20is) 來觀察 GPT-2 預測下個字的機率。
- 想要深入了解 Transformer 或 GPT-2，推薦閱讀：
  - [淺談神經機器翻譯 & 用 Transformer 與 TensorFlow 2 英翻中](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)
  - [直觀理解 GPT-2 語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)
  - [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)

---

|[NVIDIA AI PLAYGROUND](https://www.nvidia.com/en-us/research/ai-playground/)| [Grover](https://grover.allenai.org/) |
|:---:|:---:|
|<a href="https://www.nvidia.com/en-us/research/ai-playground/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/nvidia-ai-playground.jpg"></a>|<a href="https://grover.allenai.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/grover.jpg"></a>

### [NVIDIA AI PLAYGROUND](https://www.nvidia.com/en-us/research/ai-playground/)
- 提供 [GauGAN](https://arxiv.org/abs/1903.07291) 的線上展示，讓你可以利用簡單的筆觸來生成真實世界的風景圖片，也能上傳自己的圖片做風格轉換
- 提供 [Image Impainting](https://arxiv.org/abs/1804.07723) 服務，讓使用者自由抹去部分圖片並讓 AI 自動生成被抹去的區塊

### [Grover](https://grover.allenai.org/)
- 一個偵測 / 生成神經假新聞（Neural Fake News）的研究，其網頁展示如何自動生成假新聞。

---

|[Waifu Vending Machine](https://waifulabs.com)| [This Waifu Does Not Exist](https://www.thiswaifudoesnotexist.net/)
|:---:|:---:|
|<a href="https://waifulabs.com"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/waifulabs.jpg"></a>|<a href="https://www.thiswaifudoesnotexist.net/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/thiswaifudoesnotexist.jpg"></a>

### [Waifu Vending Machine](https://waifulabs.com)

- Waifu 來自日文 ワイフ，指的是一些非常受到歡迎、且被不少玩家/觀眾視為妻子的動漫女性角色。[Sizigi Studios](https://twitter.com/SizigiStudios) 團隊利用 GAN 隨機初始 16 名虛擬動漫角色，讓使用者可以進一步依照喜愛來創造專屬於自己的 Waifu。
- Waifu Vending Machine 產生的 Waifu 品質很高，使用者可以下載並分享自己創造的 Waifu，也可以選擇購買印製該 Waifu 的海報與抱枕。

### [This Waifu Does Not Exist](https://www.thiswaifudoesnotexist.net/)

- 以 Nvidia 的 [StyleGAN](https://github.com/NVlabs/stylegan) 隨機生成的 Waifu（右圖左側）。作者 [Gwern](https://www.gwern.net/) 同時也使用[開源的小型 GPT-2](https://blog.openai.com/better-language-models/) 隨機生成一段動漫劇情（右圖右側）。自釋出後已超越一百萬使用者拜訪該網站。
- 你也可以用大螢幕查看作者的另個相關網站：[These Waifus Do Not Exist](https://www.obormot.net/demos/these-waifus-do-not-exist)，用全畫面一次「觀賞」數十名隨機生成的 Waifus。

---

|[AI Notes](http://www.deeplearning.ai/ai-notes/)| To Be Updated
|:---:|:---:|
|<a href="http://www.deeplearning.ai/ai-notes/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/playground/deeplearning-ai-notes.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/general/to-be-updated.jpg"></a>

### [AI Notes](http://www.deeplearning.ai/ai-notes/)

- AI Notes 是 [吳恩達的 Deep Learning 專項課程](#deep-learning-specialization--coursera)的輔助教材，使用數學證明以及由 TensorFlow.js 建立的線上 demo 讓你可以直觀地學習[如何初始化神經網路權重](http://www.deeplearning.ai/ai-notes/initialization/)及[如何最佳化模型權重](http://www.deeplearning.ai/ai-notes/optimization/) 
- 縮圖為 [Parameter optimization in neural networks](http://www.deeplearning.ai/ai-notes/optimization/) 單元中使用不同 Optimiziers 訓練模型的線上 demo


## <div id='courses'>線上課程</div>
看完[遊玩空間](#playground)的大量實際應用，相信你已經迫不及待地想要開始學習強大的深度學習技術了。

這節列舉了一些有用的線上課程以及學習教材，幫助你掌握深度學習的基本知識（沒有特別註明的話皆為免費存取）。

另外值得一提的是，大部分課程都要求一定程度的 [Python](https://www.python.org/) 程式能力。

|[李宏毅教授的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)| [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)
|:---:|:---:|
|<a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/Hung-Yi-Lee-ml-courses.jpg"></a>|<a href="https://www.coursera.org/specializations/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/deep-learning-specification-coursera.jpg"></a>

### [李宏毅教授的機器學習 / 深度學習課程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)

- 大概是全世界最好、最完整的 Deep Learning <b>中文</b>學習資源。
- 影片內容涵蓋基本理論（約 10 小時觀看時間）一直到進階的[生成對抗網路 GAN](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C) 以及[強化學習 RL](https://zh.wikipedia.org/wiki/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)
- 你也可以從[這邊](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)看到教授的 Youtube 課程清單
- [李宏毅机器学习笔记(LeeML-Notes，簡體)](https://github.com/datawhalechina/leeml-notes) 則將教授上課的影片內容轉換為完整的筆記，方便快速瀏覽所有課程內容。

### [Deep Learning Specialization @ Coursera](https://www.coursera.org/specializations/deep-learning)

- 原 Google Brain 的[吳恩達](https://zh.wikipedia.org/wiki/%E5%90%B4%E6%81%A9%E8%BE%BE)教授開授的整個深度學習專項課程共分五堂課，從[神經網路的基礎](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)到能夠進行機器翻譯、語音辨識的[序列模型](https://www.coursera.org/learn/nlp-sequence-models)，每堂課預計 1 個月完成，收費採訂閱制
- 程式作業會交互使用 [Numpy](http://www.numpy.org/)、[Keras](https://keras.io/) 以及 [TensorFlow](https://www.tensorflow.org/) 來實作深度學習模型

---

|[Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)| [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)
|:---:|:---:|
|<a href="https://course.fast.ai/index.html"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/fast-ai.jpg"></a>|<a href="https://www.kaggle.com/learn/deep-learning"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/kaggle-learn-dl.jpg"></a>

### [Practical Deep Learning For Coders @ fast.ai](https://course.fast.ai/index.html)

- 7 週課程，一週約需安排 10 小時上課。該課程由[傑里米·霍華德](https://www.kaggle.com/jhoward)來講解深度學習，其在知名數據建模和數據分析競賽平台 [Kaggle](https://www.kaggle.com/) 維持兩年的世界第一

### [Deep Learning @ Kaggle Learn](https://www.kaggle.com/learn/deep-learning)

- 14 堂課程，主要使用 TensorFlow 實作深度學習模型
- 內容主要專注在[電腦視覺（Computer Vision）](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)以及如何應用[遷移學習（Transfer Learning）](https://en.wikipedia.org/wiki/Transfer_learning)

---

|[Elements of Artificial Intelligence](https://www.elementsofai.com/)| [MIT Deep Learning](https://deeplearning.mit.edu/)
|:---:|:---:|
|<a href="https://www.elementsofai.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/elementsofai.jpg"></a>|<a href="https://selfdrivingcars.mit.edu/deeptraffic"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/mlt-deep-learning.jpg"></a>

### [Elements of Artificial Intelligence](https://www.elementsofai.com/)

- 芬蘭最高學府[赫爾辛基大學](https://zh.wikipedia.org/wiki/%E8%B5%AB%E5%B0%94%E8%BE%9B%E5%9F%BA%E5%A4%A7%E5%AD%A6)推出的 AI 課程。此課程目的在於讓所有人都能了解 AI，不需要任何程式經驗。這堂課非常適合完全沒有接觸過深度學習或是相關領域的人
- 課程分 6 個部分，包含「何謂 AI ？」、「真實世界的 AI」、「機器學習」以及「神經網路」等章節

### [MIT Deep Learning](https://deeplearning.mit.edu/)

- 麻省理工學院推出的深度學習課程，內容包含深度學習基礎、深度強化學習以及自動駕駛相關知識。[Github Repo](https://github.com/lexfridman/mit-deep-learning) 包含了多個教學筆記本，值得參考
- 上圖是 [DeepTraffic](https://selfdrivingcars.mit.edu/deeptraffic/)，由 MIT 的研究科學家 [Lex Fridman](https://lexfridman.com/) 推出的一個深度強化學習競賽。此競賽目標是建立一個可以在高速公路上駕駛汽車的神經網路。你可以在[這裡](https://selfdrivingcars.mit.edu/deeptraffic/)看到線上 Demo 以及詳細說明

---

|[6.S191: Introduction to Deep Learning](http://introtodeeplearning.com)|[AI For Everyone](https://www.coursera.org/learn/ai-for-everyone)
|:---:|:---:|
|<a href="http://introtodeeplearning.com"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/intro-to-deeplearning-mit.jpg"></a>|<a href="https://www.coursera.org/learn/ai-for-everyone"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/ai-for-everyone.jpg"></a>

### [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)

- 麻省理工學院推出的另一堂基礎深度學習課程，介紹深度學習以及其應用。內容涵蓋機器翻譯、圖像辨識以及更多其他應用。此課程使用 Python 以及 TensorFlow 來實作作業，並預期學生具備基礎的微積分（梯度 & Chain Rule）以及線性代數（矩陣相乘）

### [AI For Everyone](https://www.coursera.org/learn/ai-for-everyone)

- Coursera 課程。[吳恩達](https://zh.wikipedia.org/wiki/%E5%90%B4%E6%81%A9%E8%BE%BE)教授在這堂簡短的課程裡頭，針對非技術人士以及企業經理人說明何謂 AI、如何建立 AI 專案以及闡述 AI 與社會的關係。此課程十分適合沒有技術背景的讀者。[從 AI For Everyone 學到的 10 個重要 AI 概念](https://leemeng.tw/10-key-takeaways-from-ai-for-everyone-course.html)則是我個人上完課後整理的心得分享。

---

|[CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)| [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
|:---:|:---:|
|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/cs224n.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/courses/cs231n.jpg"></a>

### [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

- 由[史丹佛 AI 實驗室的 Christopher Manning 教授](http://technews.tw/2018/11/21/stanford-ai-lab-christopher-manning/)從語言學、計算機科學的角度講述自然語言處理的所有必要知識，是想要打好 NLP 基礎的人不可不學的一堂課。課程約有 20 部影片，每部約長 1.5 小時。

### [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

- 由[史丹佛 Vision Lab 的李飛飛（Fei-Fei Li）教授](http://vision.stanford.edu/index.html)等人以[圖像分類](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture02.pdf)任務為軸心，講述卷積神經網路以及所有電腦視覺的相關基礎知識。這是想要學會使用（卷積）神經網路來處理圖像數據的人不可不學的一堂課。[Youtube 上有 16 部 2017 年的課程錄影](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)，每部約長 1 小時。
- 課程中也包含了不少線上展示，如[線性分類器的 loss 視覺化](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)、[kNN demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/) 以及圖像分類的 [CIFAR-10 demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)。


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

---

|[Embedding Projector](https://projector.tensorflow.org/)|[Lucid](https://github.com/tensorflow/lucid)|
|:---:|:---:|
|<a href="https://projector.tensorflow.org/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/embedding-projector.jpg"></a>|<a href="https://github.com/tensorflow/lucid"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/lucid.jpg"></a>|

### [Embedding Projector](https://projector.tensorflow.org/)

- 我們時常需要將圖片、文字轉成[高維數字向量 Embedding](https://en.wikipedia.org/wiki/Tensor) 以供神經網路處理，而 Projector 能將此高維向量投影到 2、3 維空間上方便我們理解這些數據
- Projector 網站讓你在線上探索幾個常見的資料集，但事實上你也可以[利用 Tensorboard 來視覺化自己的數據](https://www.tensorflow.org/guide/embedding)。

### [Lucid](https://github.com/tensorflow/lucid)

- Lucid 是一個嘗試讓神經網路變得更容易解釋的開源專案，裡頭包含了很多視覺化神經網路的筆記本
- 你可以直接在 Colab 上執行這些筆記本並了解如何視覺化神經網路

---

|[Papers with Code](https://paperswithcode.com/)| [What-If Tool](https://pair-code.github.io/what-if-tool/)
|:---:|:---:|
|<a href="https://paperswithcode.com/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/papers-with-code.jpg"></a>|<a href="https://pair-code.github.io/what-if-tool/"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/what-if-tool.jpg"></a>

### [Papers with Code](https://paperswithcode.com/)

- 將機器學習的學術論文、程式碼實作以及 SOTA 的評價排行榜全部整理匯總在一起的網站，非常適合想要持續追蹤學術及業界最新研究趨勢的人
- 在這邊可以瀏覽包含電腦視覺、自然語言處理等各大領域在不同任務上表現最好的論文、實作以及資料集

### [What-If Tool](https://pair-code.github.io/what-if-tool/)

- 一個與 [TensorBoard](#tensorboard) 以及 Jupyter Notebook 整合的探索工具，讓使用者不需寫程式碼就能輕鬆觀察機器學習模型的內部運作以及嘗試各種 What-if 問題（如果 ~ 會怎麼樣？）
- 基本上就是用來觀察**已訓練**的模型在測試資料集上的表現。利用此工具，使用者可以了解（不僅限於）以下的問題：模型在各類別數據上的表現有無差距？模型是否存在偏見？應該如何調整 Native / Positive False 的比例？
- 此工具的一大亮點在於讓非專業領域人士也能探索、理解 ML 模型表現。且只要給定模型與資料集, 就不需要每次為了 What-if 問題就寫用過即丟的程式碼

---

|[BertViz](https://github.com/jessevig/bertviz)| To Be Updated
|:---:|:---:|
|<a href="https://github.com/jessevig/bertviz"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tools/bertviz.jpg"></a>|<a href=""><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/general/to-be-updated.jpg"></a>

### [BertViz](https://github.com/jessevig/bertviz)

- BertViz 是一個視覺化自注意力機制的工具，可以用來理解如 [BERT](https://arxiv.org/abs/1810.04805)、[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 及 [RoBERTa](https://arxiv.org/abs/1907.11692) 等知名 NLP 模型的內部運作
- 以下則是幾篇透過 BertViz 來直觀解說 BERT 與 GPT-2 的文章
  - [進擊的 BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
  - [直觀理解 GPT-2 語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)


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
- 繁體中文的翻譯書籍則為 [Deep learning 深度學習必讀 - Keras 大神帶你用 Python 實作](https://www.tenlong.com.tw/products/9789863125501?list_name=i-r-zh_tw)
- Keras 在 TensorFlow 2.0 中[為其最重要的高層次 API](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)

---

|[Stanford CS230 Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)|[practicalAI](https://github.com/madewithml/practicalAI)|
|:---:|:---:|
|<a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/cs230-deep-learning-cheatsheet.jpg"></a>|<a href="https://github.com/GokuMohandas/practicalAI"><img src="https://github.com/leemengtaiwan/deep-learning-resources/raw/master/images/tutorials/practical-ai-pytorch.jpg"></a>

### [Stanford CS230 Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

- 史丹佛大學的[深度學習課程 CS230](http://cs230.stanford.edu/) 釋出的深度學習小抄總結了目前最新的[卷積神經網路](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)及[循環神經網路](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)知識，還包含了[訓練深度學習時需要使用到的技巧](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)，十分強大
- 此小抄最適合已經熟悉基礎知識的同學隨時複習運用。你也可以從他們的 [Github Repo](https://github.com/afshinea/stanford-cs-230-deep-learning) 下載包含上述所有內容的[超級 VIP 小抄](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/master/en/super-cheatsheet-deep-learning.pdf)
- 除了深度學習以外，你也可以查看 [CS229 機器學習課程的小抄](https://stanford.edu/~shervine/teaching/cs-229.html)

### [practicalAI](https://github.com/madewithml/practicalAI)
- 在 Github 上超過 1 萬星的 Repo。除了深度學習，也有介紹 [Python 基礎](https://colab.research.google.com/github/practicalAI/practicalAI/blob/master/notebooks/basic_ml/01_Python.ipynb)及 [Pandas](https://colab.research.google.com/github/practicalAI/practicalAI/blob/master/notebooks/basic_ml/03_Pandas.ipynb) 的使用方式
- 使用 [PyTorch](https://pytorch.org/) 框架來實現深度學習模型，且所有內容都是 Jupyter 筆記本，可以讓你在 Colab 或本地端執行

---

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
- [2017/08 Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
    - 監督式預訓練。透過 BiLSTM 與 Encoder-Decoder 架構預先訓練機器翻譯任務並將訓練後的 Encoder 拿來做特徵擷取。將 Encoder 的輸出作為語境向量（Context Vectors, CoVe）處理下游任務
- [2018/01 Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- [2018/02 Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - [ELMo 詞向量](https://allennlp.org/elmo)，利用兩獨立訓練的 LSTM 獲取雙向訊息
- [2018/06 Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - [OpenAI](https://blog.openai.com/language-unsupervised/) 利用無監督式預訓練以及 Transformer 架構訓練出來的模型表現在多個 NLP 任務表現良好。約使用 8 億詞彙量的資料集
- [2018/10 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    - Google 暴力美學。利用深層 Transformer 架構、2 個精心設計的預訓練任務以及約 33 億詞彙量的資料集訓練後，得到表現卓越的語言代表模型，打破 11 項 NLP 任務紀錄
- [2019/05 MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450)
    - Microsoft 利用 Encoder-Decoder 架構以及連續遮罩（consecutive mask）將 BERT 推廣到自然語言生成（NLG）類型任務
- [2019/05 Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
    - 預訓練階段利用不同遮罩控制 context，同時訓練雙向 LM、單向 LM 以及 Seq2Seq LM。其產生的預訓練模型可以處理 NLU 以及 NLG 任務，並在不加入外部數據的情況下打敗 BERT 在 GLUE 的紀錄


### 電腦視覺 Computer Vision (CV)
#### 類神經網路架構 Neural Network Architecture
- [1998/01 Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [2012/12 ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [2014/06 DeepFace: Closing the Gap to Human-Level Performance in Face Verification (DeepFace)](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- [2014/09 Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [2014/09 Goint deeper with convolutions (GoogLeNet)](https://arxiv.org/abs/1409.4842)
- [2014/11 Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [2015/05 U-Net: Convolutional Networks for Biomedical Image Segmentation (U-Net)](https://arxiv.org/abs/1505.04597)
- [2015/12 Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [2017/04 MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (MobileNets)](https://arxiv.org/abs/1704.04861)
- [2017/07 ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (ShuffleNet)](https://arxiv.org/abs/1707.01083)
#### 資料集 Dataset
- [2009/06 ImageNet: A Large-Scale Hierarchical Image Database (ImageNet)](http://www.image-net.org/papers/imagenet_cvpr09.pdf)
#### 物體偵測與切割 Object Detection and Segmentation
- [2013/11 Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN)](https://arxiv.org/abs/1311.2524)
- [2013/12 OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks (OverFeat)](https://arxiv.org/abs/1312.6229)
- [2015/04 Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [2015/06 Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Faster R-CNN)](https://arxiv.org/abs/1506.01497)
- [2015/06 You Only Look Once: Unified, Real-Time Object Detection (YOLO)](https://arxiv.org/abs/1506.02640)
- [2015/12 SSD: Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325)
- [2016/12 YOLO9000: Better, Faster, Stronger (YOLOv2)](https://arxiv.org/abs/1612.08242)
- [2017/03 Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [2018/04 YOLOv3: An Incremental Improvement (YOLOv3)](https://arxiv.org/abs/1804.02767)
#### 生成模型 Generative Models
- [2014/06 Generative Adversarial Networks (GAN)](https://arxiv.org/abs/1406.2661)
- [2015/13 Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/abs/1511.06434)
- [2017/01 Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875)
- [2017/03 Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/abs/1703.10593)

## <div id="collections">其他整理</div>
這邊列出其他優質的資源整理網站 / Github Repo，供你繼續探索深度學習。

### [deep-learning-ocean](https://github.com/osforscience/deep-learning-ocean)
- 整理了不少深度學習資源，但最值得參考的是數據集以及論文的分類整理。


## 待辦事項
還有不少內容正在整理，以下是目前我們打算增加的一些項目：

- 深度學習中英術語對照表
- 值得追蹤的業界 / 學界影響人物清單
- 無圖的資源列表版本
- 一些 Jupyter Notebook 範例

而我們也會持續將新資源加入如[實用工具](#tools)、[優質文章](#blogs)等列表裡頭。

## 如何貢獻
非常歡迎你一起加入改善這個 Repo，讓更多人有方向地學習 Deep Learning：）

如果你有

- 其他值得推薦的深度學習資源
- 針對此 Repo 內容的改善建議
- 其他任何你想得到的東西

都歡迎你[提出新的 Issue](https://github.com/leemengtaiwan/deep-learning-resources/issues/new) 來讓我們知道。

如果是想增加新資源的話，只附上連結也是沒有問題的，謝謝！
