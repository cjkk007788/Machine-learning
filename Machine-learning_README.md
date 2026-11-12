# Machine-learning

> 머신러닝 핵심 알고리즘을 **NumPy로 직접 구현**하며 공부하는 학습 저장소

경사하강법에서 시작해 로지스틱 회귀, 소프트맥스 회귀, 일반화 선형 모델(GLM)까지, 라이브러리에 의존하지 않고 수식을 코드로 옮기며 원리를 익히는 것을 목표로 합니다. 알고리즘 구현(`Algorithm`)과 실제 데이터에 적용한 실습 노트북(`Practice`)으로 나누어 정리했습니다.

---

## 저장소 구조

```
Machine-learning
├── Topic
│   ├── Algorithm                       # 알고리즘 구현 (.py)
│   │   ├── gradient_decent/
│   │   │   ├── basic_gradientdecent.py     # 미니배치 경사하강법 (선형회귀 + MSE)
│   │   │   └── decay_Lr_gradient.py        # 학습률 감쇠(learning rate decay) 적용
│   │   ├── logistic_regression/
│   │   │   └── basic_logisticregression.py # 이진 분류 (sigmoid + cross entropy)
│   │   ├── softmax_regression/
│   │   │   └── basic_softmax_regression.py # 다중 분류 (full batch / mini-batch)
│   │   └── GLM/
│   │       ├── Gausian.py                  # Gaussian GLM  (항등 링크)
│   │       ├── Bernoulli.py                # Bernoulli GLM (로짓 링크)
│   │       ├── Binomial.py                 # Binomial GLM
│   │       └── Poisson.py                  # Poisson GLM   (로그 링크)
│   └── Practice                        # 실습 노트북 (.ipynb)
│       ├── gradient_decent/
│       ├── logistic_regression/
│       └── softmax_regression/
└── figure_axis.py                      # matplotlib 산점도 연습용 스크립트
```

---

## 다루는 내용

### 1. 경사하강법 (Gradient Descent)
- 선형 모델 `y = Xw + b`와 MSE 손실을 직접 구현하고, 샘플 단위 gradient를 배치로 평균내어 가중치를 갱신
- 에폭마다 데이터를 셔플하는 **미니배치 방식**
- **학습률 감쇠**: 에폭이 지날수록 학습률에 `decay`를 곱해 수렴을 안정화
- 실습: Seaborn `penguins` 데이터로 날개 길이(와 부리 길이) → 몸무게 예측, 학습 과정에서 가중치와 손실이 움직이는 경로를 시각화

### 2. 로지스틱 회귀 (Logistic Regression)
- sigmoid, log-likelihood, cross entropy와 gradient를 직접 구현 (`log(0)` 방지를 위한 clipping 포함)
- 실습: scikit-learn `breast_cancer` 데이터, 표준화 후 학습하고 `sklearn.LogisticRegression`과 비교

### 3. 소프트맥스 회귀 (Softmax Regression)
- 수치 안정적인 softmax(최댓값 빼기), one-hot 인코딩, cross entropy, `dW = (P - Y)ᵀX / m`
- **full batch**와 **mini-batch** 학습을 모두 구현하고 `evaluate`로 loss와 정확도 측정
- 4개의 데이터셋으로 실습:

| 노트북 | 데이터 | 특징 | 결과 (test accuracy) |
| --- | --- | --- | --- |
| `softmax_regression_iris` | Iris | 입문용 3클래스 분류 | full 93.3% / mini-batch 96.7% |
| `softmax_regression_digits` | scikit-learn Digits (8×8) | 틀린 이미지 확인 함수 포함 | full 96.9% / mini-batch 98.1% |
| `softmax_regression_MNIST` | MNIST (28×28) | 클래스별 가중치 시각화, 혼동 행렬 | 92.4% |
| `softmax_regression_spiral` | 직접 생성한 나선형 데이터 | **Feature Engineering** 실험 | 56.7% → 93.3% → 100% |

> Spiral 실험: 선형 모델은 비선형 경계를 만들 수 없어 정확도가 낮지만, 제곱/곱/`sin` 특성을 추가하면 93.3%, 나선 구조에 맞춘 3차원 특성(`r`, `cos(u)`, `sin(u)`)을 쓰면 100%까지 올라갑니다. 특성을 더 높은 차원으로 옮겨 선형 경계로 풀 수 있다는 점을 보여주는 실습입니다.

### 4. 일반화 선형 모델 (GLM)
선형 결합 `η = Xβ + b`에 **역링크 함수**를 적용해 서로 다른 분포를 하나의 틀로 다룹니다. 어떤 GLM이든 gradient가 `Xᵀ(μ - y) / n` 꼴로 같다는 점이 핵심입니다.

| 모델 | 분포 | 링크 / 역링크 | 용도 |
| --- | --- | --- | --- |
| `GaussianGLM` | 정규분포 | 항등 | 연속값 회귀 |
| `BernoulliGLM` | 베르누이 | 로짓 / sigmoid | 이진 분류 |
| `BinomialGLM` | 이항분포 | 로짓 / sigmoid × 시행 횟수 | 성공 횟수 예측 |
| `PoissonGLM` | 푸아송 | 로그 / exp | 카운트 데이터 |

---

## 실행 환경

```bash
pip install numpy matplotlib seaborn scikit-learn jupyter
```

- Python 3.9+
- MNIST 노트북은 `fetch_openml`로 데이터를 내려받기 때문에 처음 실행할 때 인터넷 연결이 필요합니다.
- 일부 Spiral 노트북의 그래프에 한글 제목이 있어, 폰트가 없으면 글자가 깨질 수 있습니다.

### 실행 방법

```bash
git clone https://github.com/cjkk007788/Machine-learning.git
cd Machine-learning

# 알고리즘 스크립트
python Topic/Algorithm/logistic_regression/basic_logisticregression.py

# 실습 노트북
jupyter notebook Topic/Practice/
```

> `Algorithm`의 `.py` 파일들은 함수/클래스 정의 중심입니다. 실습 노트북은 필요한 함수를 노트북 안에 포함하고 있어서, 노트북만 단독으로 실행할 수 있습니다.

---

## 알려진 이슈 / TODO

- [ ] `GLM/Binomial.py` 수정 필요
  - `sigmoid(self, Z)` 안에서 `z`를 사용 (인자 이름 불일치)
  - 반환식 `1.0 / 1.0 + np.exp(-z)`의 연산자 우선순위 오류 (`1.0 / (1.0 + np.exp(-z))`이어야 함)
  - `fit`의 반복 횟수가 `n_iterations`가 아닌 `n_trials`로 되어 있음
  - `predict_proba`가 존재하지 않는 `_sigmoid`를 호출
- [ ] GLM 모델들을 실제 데이터로 검증하는 실습 노트북 추가
- [ ] `decay_Lr_gradient.py`의 두 가지 감쇠 방식(`Decay_Gradient_decent`, `Decay_Gradient_decent2`) 비교 정리
- [ ] 정규화(L1/L2), 모멘텀, Adam 등 optimizer 추가
- [ ] 폴더명 `gradient_decent` → `gradient_descent` 오타 정리
- [ ] 노트북에 개념 설명용 마크다운 셀 추가

## License

아직 라이선스가 지정되지 않았습니다.
