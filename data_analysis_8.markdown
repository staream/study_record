---
layout: post
title: 	"7th & 8th chapter of Data analyst"
date: 	2024-10-07 00:50:17 +0900
categories: KhuDa
---

수치 시뮬레이션으로 소비자 행동을 예측하는 테크닉 10

Preview

앞장은 최적화 수법을 이용해서 물류 사업의 경영 개선이 목적인 process를 설명했다.

- 이는 여러 상황에서 적용 가능하다.
- 조건 누락, 오류는 곧 다른 정답을 도출시킨다.
- 현실적인 한계는 조건의 설정이나 해의 검증을 보장할 수 없다.

수치 시뮬레이션
- 선택지를 넓히는 수법

수치 시뮬레이션 활용 방안
- 소비자 행동이 입소문과 interaction의 영향

전제 조건
- 연결을 나타내는 데이터 셋을 이용한다.
- 연결 : 이용의 여부도 해당한다.

#### 인간관계 네트워크 시각화
데이터 불러오기
```python
import pandas as pd

df_links = pd.read_csv("links.csv")
```
위의 데이터는 연결의 여부에 대한 관계를 기록
- 1 : 연결
- 0 : 연결 x

```python
import networkx as nx
import matplotlib.pyplot as plt

# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
NUM = len(df_links.index)
for i in range(1,NUM+1):
    node_no = df_links.columns[i].strip("Node")
    G.add_node(str(node_no))

# 엣지 설정
for i in range(NUM):
    for j in range(NUM):
        node_name = "Node" + str(j)
        if df_links[node_name].iloc[i]==1:
            G.add_edge(str(i),str(j))
        
# 그리기
nx.draw_networkx(G,node_color="k", edge_color="k", font_color="w")
plt.show()
```
<img src="image-7.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

사용함수
- draw가 아닌, draw_network 사용
    - 특징 : 연결이 많은 노드는 중심으로 오게 가시화한다.

#### 입소문에 의한 정보 전파 모습 가시화

```python
import numpy as np

def determine_link(percent):
    rand_val = np.random.rand()
    if rand_val<=percent:
        return 1
    else:
        return 0


def simulate_percolation(num, list_active, percent_percolation):
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                node_name = "Node" + str(j)
                if df_links[node_name].iloc[i]==1:
                    if determine_link(percent_percolation)==1:
                        list_active[j] = 1
    return list_active


# 입소문에 대한 setting
percent_percolation = 0.1
T_NUM = 36
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_percolation(NUM, list_active, percent_percolation)
    list_timeSeries.append(list_active.copy())
```
determine_link : 
- 전파 여부를 확률적으로 결정한다.
- 인수 : 입소문을 낼 확률

simulate_percolation
- 역할 : 입소문을 시뮬레이션한다.
- 인수 : 
    - num - 사람수, 
    - list_active : 각 노드에 입소문 전달됐는지를 1또는 0으로 표현한 배열, 
    - percent_percolation : 입소문을 일으킬 확률

```python

# 액티브 노드 가시화 #
def active_node_coloring(list_active):
    #print(list_timeSeries[t])
    list_color = []
    for i in range(len(list_timeSeries[t])):
        if list_timeSeries[t][i]==1:
            list_color.append("r")
        else:
            list_color.append("k")
    #print(len(list_color))
    return list_color


# 그리기
t = 0
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()

t =11
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()

t = 35
nx.draw_networkx(G,font_color="w",node_color=active_node_coloring(list_timeSeries[t]))
plt.show()
```
<img src="image-8.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>


<img src="image-9.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

<img src="image-10.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

입소문이 활성화된 노드는 빨간색으로, 전파되지 않은 노드는 검은색으로 색칠하는 함수이다.

#### 입소문 수의 시계열 변화를 그래프로

```python

# 시계열 그래프 그리기
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))
# 열에 있는 데이터를 모두 더하여 list에 넣고 있다.

plt.plot(list_timeSeries_num)
plt.show()
```

<img src="image-11.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

#### 회원 수의 시계열 변화를 시뮬레이션해 보자

분석과 시뮬레이션를 통해 미래 예측
```python
def simulate_population(num, list_active, percent_percolation, percent_disapparence,df_links):
    # 확산 #
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                if df_links.iloc[i][j]==1:
                    if determine_link(percent_percolation)==1:
                        list_active[j] = 1
    # 소멸 #
    for i in range(num):
        if determine_link(percent_disapparence)==1:
            list_active[i] = 0
    return list_active



percent_percolation = 0.1
percent_disapparence = 0.05
T_NUM = 100
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
    list_timeSeries.append(list_active.copy())


# 시계열 그래프 그리기
list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num)
plt.show()

```
simulate_population 함수
- 입소문 전파 
- '소멸' (회원 탈퇴)

<img src="image-12.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

소멸 확률을 0.2로 한 경우
<img src="image-13.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

#### 파라미터 전체를 '상관관계'를 보면서 파악
입소문의 전파와 이용 중단의 확률
- 상품이나 서비스의 성질에 따라 달라진다.
- 캠페인의 유무에 따라 달라짐.

상품의 보급
- 전파나 중단의 확률에 따라 달라짐
```python
# 상관관계 계산
print("상관관계 계산시작")
T_NUM = 100
NUM_PhaseDiagram = 20
phaseDiagram = np.zeros((NUM_PhaseDiagram,NUM_PhaseDiagram))
for i_p in range(NUM_PhaseDiagram):
    for i_d in range(NUM_PhaseDiagram):
        percent_percolation = 0.05*i_p
        percent_disapparence = 0.05*i_d
        list_active = np.zeros(NUM)
        list_active[0] = 1
        for t in range(T_NUM):
            list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_links)
        phaseDiagram[i_p][i_d] = sum(list_active)
print(phaseDiagram)
```

```python

# 표시
plt.matshow(phaseDiagram)
plt.colorbar(shrink=0.8)
plt.xlabel('percent_disapparence')
plt.ylabel('percent_percolation')
plt.xticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.yticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
plt.show()
```

<img src="image-14.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

추론
- 소멸 확률이 20~30%를 넘으면 입소문 확률이 높더라도 이용자는 증가하지 않는 모습을 볼 수 있다.

#### 실제 데이터를 불러와보자

```python
import pandas as pd

df_mem_links = pd.read_csv("links_members.csv")
# 540명 각각이 SNS 연결 여부
df_mem_info = pd.read_csv("info_members.csv")
# 24개월간 이용 현황 -> 이용 달 : 1
```

#### 링크 수의 분포 가시화
링크 수 파악의 필요성

- 이전처럼 네트워크 가시화도 가능하다.
- 규모의 측면에선 네트워크를 가시화해도
- 노드의 밀집으로 상황 파악이 어렵다.
```python
NUM = len(df_mem_links.index)
array_linkNum = np.zeros(NUM)
for i in range(NUM):
    array_linkNum[i] = sum(df_mem_links["Node"+str(i)])

plt.hist(array_linkNum, bins=10,range=(0,250))
plt.show()
```
<img src="image-15.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>
Matplotlib의 hist함수로 링크 개수의 히스토그램을 표시한다.

그래프는 링크 개수가 대략 100 정도에 집중된 정규분포에 가까운 모습이다.

스케일 프리형
- 허브(링크 많음)가 작동하지 않으면, 중간에 퍼지지 않는다.
    - 이에 대해 거의 모든 노드가 어느 정도의 링크 수를 가지고 있기에 '허브에 의존하지 않고 입소문이 퍼지기 쉽다'라고 말할 수 있다.

#### 실제 데이터로부터 파라미터 추정
추정한 확률중에서 소멸확률
- 활성이 비활성으로 변하는 비율을 세어보면 추정가능

입소문이 전파되는 확률
- 정확히 알 수 없으니, 일단 밑의 코드처럼 중복하여 세지 않게 한 것
```python
NUM = len(df_mem_info.index)
T_NUM = len(df_mem_info.columns)-1
# 소멸 확률 추정 #
count_active = 0
count_active_to_inactive = 0
for t in range(1,T_NUM):
    for i in range(NUM):
        if (df_mem_info.iloc[i][t]==1):
            count_active_to_inactive += 1
            if (df_mem_info.iloc[i][t+1]==0):
                count_active += 1
estimated_percent_disapparence = count_active/count_active_to_inactive



# 확산 확률 추정 #
count_link = 0
count_link_to_active = 0
count_link_temp = 0
for t in range(T_NUM-1):
    df_link_t = df_mem_info[df_mem_info[str(t)]==1]
    temp_flag_count = np.zeros(NUM)
    for i in range(len(df_link_t.index)):
        df_link_temp = df_mem_links[df_mem_links["Node"+str(df_link_t.index[i])]==1]
        for j in range(len(df_link_temp.index)):
            if (df_mem_info.iloc[df_link_temp.index[j]][t]==0):
                if (temp_flag_count[df_link_temp.index[j]]==0):
                    count_link += 1
                if (df_mem_info.iloc[df_link_temp.index[j]][t+1]==1):
                    if (temp_flag_count[df_link_temp.index[j]]==0):
                        temp_flag_count[df_link_temp.index[j]] = 1 
                        count_link_to_active += 1
estimated_percent_percolation = count_link_to_active/count_link
```
##### kick ? code is so hard

#### 실제 데이터와 시뮬레이션 비교
<img src="image-13.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>
<img src="image-13.png" width="px" height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

```python

percent_percolation = 0.025184661323275185
percent_disapparence = 0.10147163541419416
T_NUM = 24
NUM = len(df_mem_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_mem_links)
    list_timeSeries.append(list_active.copy())

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))


plt.plot(list_timeSeries_num, label = 'simulated')
plt.plot(list_timeSeries_num_real, label = 'real')
plt.xlabel('month')
plt.ylabel('population')
plt.legend(loc='lower right')
plt.show()
```
![alt text](image-16.png)
height="250px" title="px(픽셀) 크기 설정" alt="network_visualization"></img><br/>

위의 코드는 시뮬레이션에 의한 이용자 수와 실제 데이터가 비슷하게 움직는 것을 확인 수 있다.

주의
- 시간에 따른 
- 증가할 때는 3~5개월에 이르게 증가하거나 늦게 증가하거나 경우가 있어, 오차가 생긴다.

#### 시뮬레이션으로 미래 예측


```python

percent_percolation = 0.025184661323275185
percent_disapparence = 0.10147163541419416
T_NUM = 36
NUM = len(df_mem_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disapparence,df_mem_links)
    list_timeSeries.append(list_active.copy())

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

plt.plot(list_timeSeries_num, label = 'simulated')
plt.xlabel('month')
plt.ylabel('population')
plt.legend(loc='lower right')
plt.show()
```
![alt text](image-17.png)

24개월 이후엔 '평범한 결과'라고 보이지만, 중요한 것은 population이 급격히 할 수 있는 지

시뮬레이션은 사전에 생길수 있는 문제를 알 수 있기에
