# Backpropagation #

(https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

### Example Network Architecture ###
![image1](http://gdurl.com/yTpO)

0.05와 0.1을 입력으로 받고 0.01과 0.99를 출력하는 네트워크

### The Forward Pass ###
![math_tni_h1](https://latex.codecogs.com/gif.latex?net_%7Bh1%7D%20%3D%20i_%7B1%7D%20*%20w_%7B1%7D%20&plus;%20i_%7B2%7D%20*%20w_%7B2%7D%20&plus;%20b_%7B1%7D%20*%201)(total net input을 계산하는 법.)

![math_tni_h1_exp](https://latex.codecogs.com/gif.latex?net_%7Bh1%7D%20%3D%200.05%20*%200.15%20&plus;0.10%20*%200.2%20&plus;0.35%20*%201%20%3D%200.3775)

그 후 출력값을 얻기위해 logistic function(예를 들면 sigmoid같은..)을 이용해 출력값을 squash(압축)합니다.

![math_squash_h1](https://latex.codecogs.com/gif.latex?out_%7Bh1%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-net_%7Bh1%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-0.3775%7D%7D%20%3D%200.593269992)

h2도 똑같이 진행합니다.

![math_tni_h2](https://latex.codecogs.com/gif.latex?net_%7Bh2%7D%20%3D%20i_%7B1%7D%20*%20w_%7B3%7D%20&plus;%20i_%7B2%7D%20*%20w_%7B4%7D%20&plus;%20b_%7B1%7D%20*%201)
![math_tni_h2_exp](https://latex.codecogs.com/gif.latex?net_%7Bh2%7D%20%3D0.05%20*%200.25%20&plus;0.1*%200.3%20&plus;0.35%20*%201%20%3D%200.3925)
![math_squash_h2](https://latex.codecogs.com/gif.latex?out_%7Bh2%7D%20%3D%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-net_%7Bh2%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-0.3925%7D%7D%20%3D%200.596884378)

이 방식을 output layer neuron까지 반복합니다.
(위 쪽의 그림대로라면 o1과 o2는 hidden layer의 출력값을 입력값으로 받습니다.)

![math_tni_o1](https://latex.codecogs.com/gif.latex?net_%7Bo1%7D%20%3D%20out_%7Bh1%7D%20*%20w_%7B5%7D%20&plus;%20out_%7Bh2%7D%20*%20w_%7B6%7D%20&plus;%20b_%7B2%7D%20*%201)
![math_tni_o1_exp](https://latex.codecogs.com/gif.latex?net_%7Bo1%7D%20%3D%200.593269992%20*%200.4%20&plus;%200.596884378%20*%200.45%20&plus;0.6%20*%201%20%3D%201.105905967)
![math_squash_o1](https://latex.codecogs.com/gif.latex?out_%7Bo1%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-net_%7Bo1%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-1.105905967%7D%7D%20%3D%200.75136507)

똑같은 방식으로 o2도 진행합니다.
![math_tni_o2](https://latex.codecogs.com/gif.latex?net_%7Bo2%7D%20%3D%20out_%7Bh1%7D%20*%20w_%7B7%7D%20&plus;%20out_%7Bh2%7D%20*%20w_%7B8%7D%20&plus;%20b_%7B2%7D%20*%201)
![math_tni_o2_exp](https://latex.codecogs.com/gif.latex?net_%7Bo2%7D%20%3D%200.593269992%20*0.5&plus;%200.596884378%20*%200.55%20&plus;%200.6%20*%201%20%3D%201.2249214039)
![math_squash_o2](https://latex.codecogs.com/gif.latex?out_%7Bo2%7D%20%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-net_%7Bo2%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-1.2249214039%7D%7D%20%3D%200.772928465)

The Forward Pass의 모든 계산을 완료했습니다!!
실제로 현재 모델에 0.05와 0.1을 입력값으로 주면 출력값은 0.75136507과 0.772928465가 나오게 됩니다.
### 총 에러율 계산 ###
이제 squared(제곱) error function을 이용해 각각의 출력의 에러를 계산합니다. 그리고 이것들을 합쳐 총 에러율을 구합니다.

![math_E](https://latex.codecogs.com/gif.latex?E_%7Btotal%7D%20%3D%20%5Csum%20%5Cfrac%7B1%7D%7B2%7D%28target-output%29%5E%7B2%7D)

 예시로 우리가 원하는 저 식을 바탕으로 각각의 에러를 구하면 다음과 같습니다.

 ![math_E1_exp](https://latex.codecogs.com/gif.latex?E_%7Bo1%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo1%7D%20-%20output_%7Bo1%7D%29%5E%7B2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%280.01%20-%200.75136507%29%5E%7B2%7D%20%3D%200.274811083)
 ![math_E2_exp](https://latex.codecogs.com/gif.latex?E_%7Bo2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo2%7D%20-%20output_%7Bo2%7D%29%5E2%20%3D%20%5Cfrac%7B1%7D%7B2%7D%280.99%20-%200.772928465%29%5E%202%20%3D0.023560026)

 그리고 이 둘을 더한것이 신경망의 총 에러률입니다.

 ![math_E_exp](https://latex.codecogs.com/gif.latex?E_%7Btotal%7D%20%3D%20E_%7Bo1%7D%20&plus;%20E_%7Bo2%7D%20%3D%200.274811083%20&plus;%200.023560026%20%3D%200.298371109)

이것으로 총 에러율 계산을 완료했습니다! 현재 신경망의 에러율은 0.298371109입니다.

### The Backwrads Pass ###
back propagation읉 통한 우리의 목표는 신경망안의 각각의 weights를 업데이트해 신경망의 출력을 우리가 원하느 출력에 가깝게 하는것 입니다.

##### output layer neurons #####
w5를 생각해 봅시다. 우리는 w5가 얼마나 error에 영향을 주는지 알고싶습니다.(w5가 1변할때 error의 변화율)

chain rule을 적용하면 다음과 같습니다.

![math_chain_rule](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D)

시각화를 하면 다음과 같습니다.

![image2](http://gdurl.com/UWJs)

이제 우리가 해야 할 일은 각각의 방정식을 찾아내는 것 입니다.

일단 앞에서부터 차례차례 진행하면 o1은 Etotal에 얼마나 영향을 끼칠 수 있을까요?

![math_E_exp](https://latex.codecogs.com/gif.latex?E_%7Btotal%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo1%7D%20-%20output_%7Bo1%7D%29%5E2%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo2%7D%20-%20output_%7Bo2%7D%29%5E2)

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20%3D%202%20*%20%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo1%7D%20-%20out_%7Bo1%7D%29%5E%7B2%20-%201%7D%20*%20-1%20&plus;%200)

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20%3D%20-%28target_%7Bo1%7D%20-%20out_%7Bo1%7D%29%20%3D%20-%280.01%20-%200.75136507%29%20%3D%200.74136507)

다음으로 out_o1이 net_o1에 미치는 영향을 알아봅시다.
(out_o1에서 squash함수를 한번 써주므로 미분을 또 한다. 한마디로 out_o1과 net_o1은 다른 함수)
[logistic function의 미분](https://en.wikipedia.org/wiki/Logistic_function#Derivative)은 다음과 같습니다.

![math](https://latex.codecogs.com/gif.latex?out_%7Bo1%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-net_%7Bo1%7D%7D%7D)

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20%3D%20out_%7Bo1%7D%281%20-%20out_%7Bo1%7D%29%20%3D%200.75136507%281%20-0.75136507%20%29%20%3D%200.186815602)

마지막으로 w5가 net_o1에 미치는 영향을 알아봅시다.

![math](https://latex.codecogs.com/gif.latex?net_%7Bo1%7D%20%3D%20out_%7Bh1%7D%20*%20w_%7B5%7D%20&plus;%20out_%7Bh2%7D%20*w_%7B6%7D%20&plus;%20b_%7B2%7D%20*%201)

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%201%20*%20out_%7Bh1%7D%20&plus;%20w_%7B5%7D%5E%7B1%20-1%7D%20&plus;%200%20&plus;%200%20%3D%20out_%7Bh1%7D%20%3D%200.593269992)

이제 모두 곱하면 다음과 같습니다.

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D)

![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%200.74136507%20*%200.186815602*%200.593269992%20%3D%200.082167041)


***
```
아마 당신은 이러한 계산을 delta rule의 형태로 결합된 계산을 자주 보게 될 것 입니다.
```
>![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20-%28target_%7Bo1%7D%20-%20out_%7Bo1%7D%29%20*%20out_%7Bo1%7D%281%20-%20out_%7Bo1%7D%29%20*%20out_%7Bh1%7D)
```
또는 다음과 같이 쓸 수 있습니다. 저러한 식을 일명 δo1(delta o1)이라 칭합니다.
```
>![math](https://latex.codecogs.com/gif.latex?%5Cdelta%20o_%7B1%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D)

>![math](https://latex.codecogs.com/gif.latex?%5Cdelta%20o_%7B1%7D%20%3D%20-%28target_%7Bo1%7D%20-%20out_%7Bo1%7D%29%20*%20out_%7Bo1%7D%281%20-%20out_%7Bo1%7D%29)

>![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20%5Cdelta%20o_1%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20%5Cdelta%20o_1%20out_%7Bh1%7D)
```
몇몇의 소스에서는 델타기호에서 음수 부호를 추출합니다. 따라서 다음과 같이 작성해야 합니다.
```
>![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%20-%5Cdelta%20o_1%20out_%7Bh1%7D)
***

에러를 줄이기 위해 우리는 현재 값에서 이 값을 빼야 합니다.(임의의 학습속도 η를 곱한다.)

![math](https://latex.codecogs.com/gif.latex?w_%7B5%7D%5E%7B&plus;%7D%20%3D%20w_%7B5%7D%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%3D%200.4%20-%200.5%20*%200.082167041%20%3D%200.35891648)

w6, w7, w8의 새로운 값을 얻기위해 이 방식을 반복합니다.

* W6
>![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B6%7D%7D%20%3D%20-%5Cdelta_%7Bo1%7Dout_%7Bh2%7D%20%3D%20-%28target_%7Bo1%7D%20-%20out_%7Bo1%7D%29%20*%20out_%7Bo1%7D%281%20-%20out_o1%29%20*%20out_%7Bh2%7D)
>
>![math](https://latex.codecogs.com/gif.latex?-%280.01%20-%200.75136507%29%20*%200.75136507%281%20-%200.75136507%29%20*%200.596884378%3D0.08266762776)
>
>![math](https://latex.codecogs.com/gif.latex?w_%7B6%7D%5E%7B&plus;%7D%20%3D%20w_%7B6%7D%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B6%7D%7D%20%3D%200.45%20-%200.5%20*%200.08266762776%20%3D%200.408666186)

* W7
>![math](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B7%7D%7D%20%3D%20-%5Cdelta_%7Bo2%7Dout_%7Bh1%7D%20%3D%20-%28target_%7Bo2%7D%20-%20out_%7Bo2%7D%29%20*%20out_%7Bo2%7D%281%20-%20out_%7Bo2%7D%29%20*%20out_%7Bh1%7D)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20-%280.99%20-%200.772928465%29%20*%200.772928465%281%20-%200.772928465%29%20*%200.593269992%20%3D%20-0.02260254053)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_%7B7%7D%5E%7B&plus;%7D%20%3D%20w_%7B7%7D%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B7%7D%7D%20%3D%200.5%20-%200.5%20*%20-0.02260254053%20%3D%200.511301270)

* W8
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B8%7D%7D%20%3D%20-%5Cdelta_%7Bo2%7Dout_%7Bh2%7D%20%3D%20-%28target_%7Bo2%7D%20-%20out_%7Bo2%7D%29%20*%20out_%7Bo2%7D%281%20-%20out_%7Bo2%7D%29%20*%20out_%7Bh2%7D)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20-%280.99%20-%200.772928465%29%20*%200.772928465%281%20-%200.772928465%29%20*%200.596884378%20%3D%20-0.02274024226)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_%7B8%7D%5E%7B&plus;%7D%20%3D%20w_%7B8%7D%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btoatl%7D%7D%7B%5Cpartial%20w_%7B8%7D%7D%20%3D%200.55%20-%200.5%20*%20-0.02274024226%20%3D%200.561370121)

hidden layer neurons의 새로운 weights를 얻은 후에 신경망 업데이트를 수행합니다.

##### hidden layer neurons #####
이제 계속해서 새로운 w1, w2, w3, w4를 계산합니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20*%20%5Cfrac%7B%20%5Cpartial%20out_%7Bh1%7D%20%7D%7B%5Cpartial%20net_%7Bh1%7D%7D%20*%20%5Cfrac%7B%20%5Cpartial%20net_%7Bh1%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D)

![image3](http://gdurl.com/xgPE)

output layer neurons에서 진행했던 방식과 비슷하지만 약간 다른 방식을 사용 할 것 입니다. 왜냐하면 h1의 출력물은 o1과 o2모두에 영향을 끼치므로 미분값도 두 output neurons의 영향을 고려할 필요가 있습니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D)

앞부터 차례대로 미분을 합니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Bo1%7D%20%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Bo1%7D%20%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20%3D%200.74136507%20*%200.186815602%20%3D%200.138498562)

다음...
![math](https://latex.codecogs.com/gif.latex?%5Csmall%20net_%7Bo1%7D%20%3D%20out_%7Bh1%7D%20*%20w_%7B5%7D%20&plus;%20out_%7Bh2%7D%20*%20w_%7B6%7D%20&plus;%20b_%7B2%7D%20*%201)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20w_%7B5%7D%20%3D0.4)

이것들을 연결합니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20net_%7Bo1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%200.138498562%20*%200.4%20%3D%200.055399425)


똑같이 두번째 식도 진행합니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20net_%7Bo2%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20net_%7Bo2%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bo2%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo2%7D%7D%7B%5Cpartial%20net_%7Bo2%7D%7D%20%3D%20-0.217071535%20*%200.175510053%20%3D%20-0.0380982366)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20net_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20w_%7B7%7D%20%3D%200.5)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20net_%7Bo2%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20-0.0380982366%20*%200.5%20%3D%20-0.019049119)

따라서 두개를 합치면 우리가 원하는 식이 완성됩니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Bo1%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20E_%7Bo2%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20%3D%200.055399425%20&plus;%20-0.019049119%20%3D%200.036350306)

이제 나머지 두가지를 찾아봅시다!

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20out_%7Bh1%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-net_h1%7D%7D) 이므로 ![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20out_%7Bh1%7D%7D%7B%5Cpartial%20net_%7Bh1%7D%7D)는 다음과 같습니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20out_%7Bh1%7D%7D%7B%5Cpartial%20net_%7Bh1%7D%7D%20%3D%20out_%7Bh1%7D%281%20-%20out_%7Bh1%7D%29%20%3D%200.59326999%20%281%20-%200.59326999%29%20%3D%200.241300709)

마지막으로 ![math](https://latex.codecogs.com/gif.latex?%5Csmall%20net_%7Bh1%7D%20%3D%20i_1%20*%20w_1%20&plus;%20i_2%20*%20w_2%20*%20b_1%20*%201) 이므로 다음과 같은 식이 나옵니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20net_%7Bh1%7D%7D%7B%5Cpartial%20w_1%7D%20%3D%20i_1%20%3D%200.05)

이제 이 모든 것을 합치면 다음과 같습니다.

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bh1%7D%7D%7B%5Cpartial%20net_%7Bh1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bh1%7D%7D%7Bw_1%7D)

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%200.036350306%20*%200.241300709%20*%200.05%20%3D%200.000438568)

***
```
위 식은 다음과 같이 변경이 가능합니다.
```
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%20%28%5Csum_%7Bo%7D%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20out_%7Bo%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bo%7D%7D%7B%5Cpartial%20net_%7Bo%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bo%7D%7D%7B%5Cpartial%20out_%7Bh1%7D%7D%29%20*%20%5Cfrac%7B%5Cpartial%20out_%7Bh1%7D%7D%7B%5Cpartial%20net_%7Bh1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20net_%7Bh1%7D%7D%7B%5Cpartial%20w_1%7D)
 >
 >![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%20%28%5Csum_%7Bo%7D%20%5Cdelta_o*%20w_%7Bho%7D%29%20*%20out_%7Bh1%7D%281%20-%20out_%7Bh1%7D%29%20*%20i_1)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%20%5Cpartial%20E_%7Btotal%7D%20%7D%7B%5Cpartial%20w_%7B1%7D%7D%20%3D%20%5Cdelta_%7Bh1%7Di_1)

***

드디어 우리는 w1을 업데이트 할 수 있습니다!!

![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_1%5E&plus;%20%3D%20w1%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_1%7D%20%3D%200.15%20-%200.5%20*%200.000438568%20%3D%200.149780716)

똑같이 반복해서 w2, w3, w4도 구해줍니다!

* W2
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_%7B2%7D%7D%20%3D%20%5Cdelta_%7Bh1%7Di_2%20%3D%200.036350306%20*%200.241300709%20*%200.1%20%3D%200.000877135461)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_2%5E&plus;%20%3D%20w_2%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7Bw_2%7D%20%3D%200.2%20-%200.5%20*0.000877135461%20%3D%200.19956143)

* W3
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_3%7D%20%3D%20%5Cdelta_%7Bh2%7Di_1%20%3D%200.04137032256%20*%200.2406134173%20*%200.05%20%3D%200.0004977127343)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_3%5E&plus;%20%3D%20w_3%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7Bw_3%7D%20%3D%200.25%20-%200.5%20*%200.0004977127343%20%3D%200.24975114)

* W4
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_4%7D%20%3D%20%5Cdelta_%7Bh2%7Di_2%20%3D%200.04137032256%20*%200.2406134173%20*%200.1%20%3D%200.0009954254686)
>
>![math](https://latex.codecogs.com/gif.latex?%5Csmall%20w_4%5E&plus;%20%3D%20w_4%20-%20%5Ceta%20*%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7Bw_4%7D%20%3D%200.3%20-%200.5%20*%200.0009954254686%20%3D%200.29950229)


드디어 우리의 모든 weights를 업데이트 했습니다!

원래의 입력인 0.05와 0.1로 feed forward했을 때 신경망의 총 에러률은 0.298371109였지만 backpropagation을 한번 시행한 후 총 에러률은 0.291027924로 낮아졌습니다.

이것이 별로 많아 보이지는 않지만 10000번을 반복하면 에러는 0.000035085가 됩며 이 때 0.05와 0.1로 feed forward를 진행하면 출력으로 0.015912196과 0.9984065734가 생성됩니다.

[https://github.com/mattm/simple-neural-network/blob/master/neural-network.py](https://github.com/mattm/simple-neural-network/blob/master/neural-network.py)
