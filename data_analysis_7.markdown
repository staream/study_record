---
layout: post
title: 	"7th & 8th chapter of Data analyst"
date: 	2024-10-07 00:50:17 +0900
categories: KhuDa
---

6장에서 배울 내용

1. 몇 개의 라이브러리를 이용한 최적화 계산
2. 네트워크 가시화 기술로 타당성 확인하는 방법

우리가 다룰 대상

아래와 같이 전체적인 것을 다루기 위해선

운송 최적화뿐만 아니라 네트워크 전체의 최적화가 필요하다.

<img src="image.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network"></img><br/>

고려 대상
- 제품을 판매하는 대리점
- 판매되는 상품군
- 수요량을 근거로 공장의 생산량
- 운송비, 제고 비용

#### 운송 최적화
라이브러리 : pulp, ortoolpy.

pulp : 최적화 모델을 작성
ortoolpy. : 목적함수 생성, 최적화 문제를 품

```python
import numpy as np
import pandas as pd
from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min, addvars, addvals

# 데이터 불러오기
df_tc = pd.read_csv('trans_cost.csv', index_col="공장")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 초기 설정  #
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
pr = list(product(range(nw), range(nf)))

# 수리 모델 작성  #
m1 = model_min()
v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr}

m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr)
for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]
m1.solve()

# 총 운송 비용 계산#
df_tr_sol = df_tc.copy()
total_cost = 0
for k,x in v1.items():
    i,j = k[0],k[1]
    df_tr_sol.iloc[i][j] = value(x)
    total_cost += df_tc.iloc[i][j]*value(x)
    
print(df_tr_sol)
print("총 운송 비용:"+str(total_cost))


```

```python
m1 = model_min()
```
최소화를 실행하는 모델로 m1을 정의하고 있다.


v1
- LpVariable을 사용해서 dict로 정의 
```python
v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr}

```
목적함수m1 정의
- lpSum을 이용해서 정의
- 운송 결로의 비용 데이터(df_tc)와 v1과 요소의 곱의 합으로 정의
```python
m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr)
```

제약 조건 m1 정의
- 제약 조건은 lpSum을 통해 정의한다.
- 제약 조건 : 
    - 공장이 제조할 제품 소요량을 만족 
    - 창고가 제공할 부품이 제공 한계를 넘지 않게 함. 
```python
for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]
```

```python   
m1.solve() # 최적화 문제를 해결하는 함수 .solve()
# solve는 변수 v1이 최적화되고, 최적의 총 운송 비용이 구한다.
```

#### 최적 경로 시각화

기본 밑작업

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 데이터 불러오기
df_tr = df_tr_sol.copy()
df_pos = pd.read_csv('trans_route_pos.csv')
```

Graph 불러오기

```python
# 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])
# df_pos의 열에 나와있는 것을 node화 시킨다.

# 엣지 설정 & 엣지의 가중치 리스트화
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i==j):
            # 엣지 추가
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # 엣지 가중치 추가
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)
                

# 좌표 설정
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])
    
# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()
```
<img src="image-1.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

추론
1. 특정 네트워크만이 활발한 교류를 보인다

##### kick!! of visualization
- 그래프를 다시 추가하는 방법에 대한 복습
- 엣지 설정 시 가중치의 리스트화의 기준이 어떻게 되는지가 중요할 것 같다.

#### plausible of constraints about optimal transport way
제약 조건을 계산하는 함수를 이용해서 계산된 운송 경로가 제약 조건을 만족하는지 확인해보자.
```python
import pandas as pd
import numpy as np

# 데이터 불러오기
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')

# 제약조건 계산함수
# 수요측
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 공급측
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("수요 조건 계산 결과:"+str(condition_demand(df_tr_sol,df_demand)))
print("공급 조건 계산 결과:"+str(condition_supply(df_tr_sol,df_supply)))
```
결과는 수요, 공급 모두가 제약 조건이 1로 충족되고 있음을 알 수 있다.

목적함수/제약조건의 장점
- 손을 할 필요없이 간단히 비용을 계산할 수 있다.

#### 운송이 아닌 생산 계획 데이터 부르기
물류 네트워크를 위해선 운송뿐만 아니라 생산 계획도 중요한 요소이다.
![alt text](image-2.png)


```python
import pandas as pd

df_material = pd.read_csv('product_plan_material.csv', index_col="제품")
print(df_material)
df_profit = pd.read_csv('product_plan_profit.csv', index_col="제품")
print(df_profit)
df_stock = pd.read_csv('product_plan_stock.csv', index_col="항목")
print(df_stock)
df_plan = pd.read_csv('product_plan.csv', index_col="제품")
print(df_plan)

```
<img src="image-3.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

product_plan_material.csv 의 특징 
- 두 종류의 제품
- 해당 제품의 원료의 비율

product_plan_profit.csv의 특징
- 각 제품의 이익

product_plan_stock.csv의 특징
- 현재 각 원료의 재고

product_plan.csv
- 제품의 생산량

#### 이익을 계산하는 함수 만들기
생산계획 최적화(생산 최적화)의 흐름
- 목적함수/ 제약조건 정의
- 제약 조건 아래서 목적 함수를 최소화하는 **변수 조합**을 찾기

- 이익은 이익을 계산할 함수(목적함수)
- 최대화하는 변수를 찾기
```python
# 이익 계산 함수
def product_plan(df_profit,df_plan):
    profit = 0
    for i in range(len(df_profit.index)):
        for j in range(len(df_plan.columns)):
            profit += df_profit.iloc[i][j]*df_plan.iloc[i][j]
    return profit

print("총 이익:"+str(product_plan(df_profit,df_plan)))
```
생산 계획의 총 이익은

SUM(이익 * 제조량)

#### 생산 최적화 문제를 풀기


<img src="image-1.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

```python
import pandas as pd
from pulp import LpVariable, lpSum, value
from ortoolpy import model_max, addvars, addvals


df = df_material.copy()
inv = df_stock
```
```python
m = model_max()#model_max()를 선언하여 '최대화'계산 준비


v1 = {(i):LpVariable('v%d'%(i),lowBound=0) for i in range(len(df_profit))}
m += lpSum(df_profit.iloc[i]*v1[i] for i in range(len(df_profit)))
# v1을 제품 수와 같은 차원으로 정의하고, 변수 v1과 제품별 이익의 곱의 합으로 목적함수를 정의한다.


for i in range(len(df_material.columns)):
    m += lpSum(df_material.iloc[j,i]*v1[j] for j in range(len(df_profit)) ) <= df_stock.iloc[:,i]
# 각 원료의 사용량이 재고를 넘지 않게 제약 조건을 정의한다.

m.solve()# 최적화 문제를 푼다.


df_plan_sol = df_plan.copy()
for k,x in v1.items():
    df_plan_sol.iloc[k] = value(x)
print(df_plan_sol)
print("총 이익:"+str(value(m.objective)))

```

#### 생산 계획이 제약 조건을 만족하는지 확인하자
최적화 문제의 주의점
- 결과를 이해하지 않고, 받아들이는 점
    - 실제 조건이 바뀌면 기대효과를 얻을 수 없다.
    - 목적함수와 제약 조건이 현실과 다른 경우
이해하는 방법
- 제약 조건으로 규정한 것을 알아보기
    - '각 원료의 사용량' 
    - '재고를 효율적으로 이용하는가'

```python
# 제약 조건 계산 함수
def condition_stock(df_plan,df_material,df_stock):
    flag = np.zeros(len(df_material.columns))
    for i in range(len(df_material.columns)):  
        temp_sum = 0
        for j in range(len(df_material.index)):  
            temp_sum = temp_sum + df_material.iloc[j][i]*float(df_plan.iloc[j])
        if (temp_sum<=float(df_stock.iloc[0][i])):
            flag[i] = 1
        print(df_material.columns[i]+"  사용량:"+str(temp_sum)+", 재고:"+str(float(df_stock.iloc[0][i])))
    return flag

print("제약 조건 계산 결과:"+str(condition_stock(df_plan_sol,df_material,df_stock)))
```
<img src="image-4.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

결과는 제약 조건은 모두 충족됐으며 원료2와 원료3은 재고를 모두 사용한 것을 알 수 있다.

#### 물류 네트워크 설계 문제(Kick)
다룰 문제
- 물류 네트워크 = 운송 경로 + 생산 계획 최적화 문제

고려 사항
- 판매장소
- 판매되는 상품군
- 수요량에 따른 생산량
- 운송 비용과 제조 비용의 수요를 만족하면서 최소가 되게 정식화한다.

목적함수
- 운송 비용과 제조 비용의 합

제약조건
- 각 대리점의 판매 수가 수요 수를 넘는 것으로 정의

사용 도구
- 라이브러리 : ortoolpy
- 함수 : logistics_network
```python
import numpy as np
import pandas as pd

제품 = list('AB')
대리점 = list('PQ')
공장 = list('XY')
레인 = (2,2)

# 운송비 #
tbdi = pd.DataFrame(((j,k) for j in 대리점 for k in 공장), columns=['대리점','공장'])
tbdi['운송비'] = [1,2,3,1]
print(tbdi)

# 수요 #
tbde = pd.DataFrame(((j,i) for j in 대리점 for i in 제품), columns=['대리점','제품'])
tbde['수요'] = [10,10,20,20]
print(tbde)

# 생산 #
tbfa = pd.DataFrame(((k,l,i,0,np.inf) for k,nl in zip (공장,레인) for l in range(nl) for i in 제품), 
                    columns=['공장','레인','제품','하한','상한'])
tbfa['생산비'] = [1,np.nan,np.nan,1,3,np.nan,5,3]
tbfa.dropna(inplace=True)
tbfa.loc[4,'상한']=10
print(tbfa)

from ortoolpy import logistics_network
_, tbdi2, _ = logistics_network(tbde, tbdi, tbfa,dep = "대리점", dem = "수요",fac = "공장",
                                prd = "제품",tcs = "운송비",pcs = "생산비",lwb = "하한",upb = "상한")

print(tbfa)
print(tbdi2)

```

이에 대한 결과를 통해 생산표에 
- ValY라는 항목생성-> 최적 생산량이 저장
- 운송 비표에 ValX라는 항목생성-> 최적 운송량이 저장

##### Kick
이 부분에 대해 다시 공부해보기

#### 최적 네트워크의 운송 비용과 그 내역을 계산

```python

tbdi2 = tbdi2[["공장","대리점","운송비","제품","VarX","ValX"]]
tbdi2
```
![alt text](image-5.png)

내역 해석
1. 운송비용 : 80만원
2. 경로 사용 : 운송비가 적은 X->대리점 P, 공장 Y-> 대리점 Q의 경로를 사용 

#### 최적 네트워크의 생산 비용과 그 내역을 계산하자
<img src="image-1.png" width="30px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

```python
product_cost = 0
for i in range(len(tbfa.index)):
    product_cost += tbfa["생산비"].iloc[i]*tbfa["ValY"].iloc[i]
print("총 생산비:"+str(product_cost))
```
![alt text](image-6.png)

업무 개선에 대해 이야기

