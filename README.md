# forked version of tensorflow-seq2seq-autoencoder for study
  - modified
    - modified for tensorflow 1.4.0
    - add inference.py
    - additional comments
    ![seq2seq_autoencoder](https://raw.githubusercontent.com/dsindex/blog/master/images/seq2seq_autoencoder.jpeg)
  - train
  ```
  * for example, if we have a tokenized corpus `news-sent-morph-100000.txt`.
  * `news-sent-morph-100000.txt` looks like(morph-based tokenized string) :
     ...
     슬기 롭다 게 대처 하다 ㄹ 수 있다 는 사람 만 이 승리 하다 ㄹ 수 있다 겠 다
     3 , 7 , 10 월 생 자신 을 따르다 ㄴ다고 무조건 만나다 거나 경거망동 하다 지 말다 라
     한순간 의 실수 로 후회 하다 ㄹ 일 생기다 ㄴ다
     북 , 동쪽 사람 조심 ▶ 소띠 = 친지 간 에 덕이 없다 어 베풀다 고도 원망 만 받다 는다
     ...

  $ CUDA_VISIBLE_DEVICES=7 python train.py --data-path data/news-sent-morph-100000.txt --model-path train --max-step=1000 --vocab-size=10000
  ```
  - inference
  ```
  $ CUDA_VISIBLE_DEVICES=7 python inference.py --model-path train --vocab-path data/news-sent-morph-100000.txt.vocab10000.txt --vocab-size=10000 --test-path data/news-sent-morph-100.txt
  ```

----

# seqseq-autoencoder
This is a simple seqseq-autoencoder example of tensorflow-0.9

tensorflow中的机器翻译示例代码在我看来并不是一个很好的seq2seq例子，它用的一些包装好的函数并没有简化事情，反倒让自己对初学者来说很难理解。于是我写了一个用tied-seq2seq(编码和解码用同一个神经网络)做短句子自编码器的例子，尽量简单，尽量注释。

The nueral tranlation example in trensorflow can hardly be called a good example of seqence-to-sequence model. The functions it uses make tensorflow beginers like me puzzled. So I write a tie-seq2seq (which means the encoder and decoder use the same rnn network) short text auto encoder example, with simple structure and detailed comments. 


##1.Data
There is a chinese address dataset in data/address.txt, you can play with it, or you can use any whitespace-splited data with each token a single word and each line a single sequence.


##2.How to run
python train.py --data-path data/address.txt --model-path train

for more options: python train.py -h














