# Phrase-Based & Neural Unsupervised Machine Translation
## Abstract
* 본 논문에서는 large monolingual corpora만 있을 때 이를 번역하는 방법을 다룬다.
* Neural 모델과 Phrase-based 모델 두 가지 모델을 다룬다. 

## 1. Introduction
* 기계 번역에서 large parallel corpora에 크게 의존하지 않게 하는 것이 주요 과제이다. 대다수의 언어 쌍에는 병렬 데이터가 거의 없기 때문이다. 
    * 이를 해결하기 위해서는 학습 알고리즘이 monolingual data를 잘 활용하는 것이 중요하다.
* monolingual data의 사용을 연구한 두 가지 (Lample et al., 2018; Artetxe et al., 2018)의 공통 원칙을 파악했다.
![MT](https://user-images.githubusercontent.com/79077316/127768041-687803cc-2e14-4397-bb9d-c8eaea622d6f.png)
    1. 유추된 bilingual 사전으로 MT 시스템을 초기화한다.
    2. sequence-to-sequence system 학습을 통해 언어 모델을 denoising autoencoder로 사용한다.
    3. back-translation을 통해 unsupervised 문제를 supervised 문제로 바꾼다.
    4. encoder에 의해 생성된 latent representation이 두 언어에 걸쳐 공유되도록 제한한다.

## 2. Principles of Unsupervised MT
* unsupervised MT에 1) 번역 모델의 적절한 초기화, 2) 언어 모델링 및 3) 반복적인 back-translation이 세 가지를 활용할 것이다.

### 1) Initialization
* 단어, 짧은 문장, 서브워드 등이 정렬(align)되도록 모델을 초기화한다.

### 2) Language Modeling
* source 언어와 target 언어 모두 사용하여 LM을 학습시킬 수 있다. 

### 3) Iterative Back-translation
* Back-translation은 monolingual data를 이용한 unsupervised setting을 supervised setting으로 바꿔준다.
![MT-algorithm](https://user-images.githubusercontent.com/79077316/127768548-dc89d012-9475-4d8e-97fe-6096a91facdc.png)

## 3. Unsupervised MT systems
* 이 논문은 위와 같은 원리를 따르면서 기존 Unsupervised NMT의 구조나 손실 함수를 조금 더 단순화시킨 NMT모델과 동일한 원리를 적용한 pharase-based statistical machine translation (PBSMT) 시스템을 제안한다.

### 3-1. Unsupervised NMT
1. initialization
* 관련성이 있는 언어들간에 적합한 BPE를 이용한 방법을 제안한다. 
* monolingual corpora들을 합친 코퍼스에 BPE를 적용하여 토크나이즈시키고, 이 토큰들의 임베딩을 학습하는 형태로 초기화가 이루어진다.

2. language model
* NMT에서는 denoising autoencoding을 통해 loss를 최소화하는 방식으로 언어 모델을 생성한다.
    
    ![MT-loss](https://user-images.githubusercontent.com/79077316/127768846-80cfe33f-3678-48a7-800f-7c56c3cf607d.png)

3. back-translation
* `u*(y)`와 `v*(x)`를 각각 역번역 모델, 번역 모델로 추론된 noisy한 문장이라고 하면, back-translation loss는 다음과 같다.
    
    ![MT-loss2](https://user-images.githubusercontent.com/79077316/127768910-d9c0d203-5ace-4229-82ea-dde3d523b890.png)

4. sharing latent representations
* 입력 source language에 상관없이 encoder representation를 공유하도록 한다. 
    * 이는 디노이징 오토인코더를 통한 언어 모델의 지식이 noisy source sentences로부터의 번역에 잘 전이되도록 도와준다. Encoder representations를 공유하기 위해서 인코더 파라미터를 공유한다.

### 3-2. Unsupervised PBSMT
![MT-algorithm2](https://user-images.githubusercontent.com/79077316/127769025-b85991d5-e3b9-488f-af1c-69db783d3fc3.png)

1. initialization
* inferred bilingual dictionary를 이용하여 초기의 phrase tables를 수집한다.

2. language model
* KenLM (Heafield, 2011)을 이용하여 n-gram 언어 모델을 학습한다.

3. back-translation
* unigram phrase tables와 target  언어 모델을 이용하여 seed PBSMT 모델을 만든다. 
* 이 모델을 이용하여 source monolingual corpus를 target language로 번역하여 대역 데이터를 생성한다. 
* 그 후, 생성된 데이터를 이용하여 supervised한 방식으로 PBSMT 모델을 학습하고, 데이터의 생성과 학습을 반복해나가면서 번역 성능을 향상시킨다.

## 4. Experiments
생략 
