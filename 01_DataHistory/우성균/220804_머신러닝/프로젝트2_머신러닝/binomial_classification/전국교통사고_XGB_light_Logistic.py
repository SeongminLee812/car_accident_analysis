#%%


# 데이터 핸들링 라이브러리
import pandas as pd
import numpy as np
# 데이터 시각화 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({'font.family':'AppleGothic'})
mpl.rc('axes', unicode_minus=False)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


#%%
# ## 데이터 전처리 계획
# 1. 데이터 불러오기, 결측치 처리하기
# 2. 상대적으로 중요성이 떨어져보이는 피처 삭제, 필요하면 피처명을 직관적으로 다시 네이밍하기
# 3. 비숫자-문자열 독립변수 인코딩하기
# 4. 데이터 정규화
# 5. 훈련-테스트셋 분리
#
# ## 모델 학습 계획
# 1. 모델 생성
# 2. Cross_validation을 통한 모델학습
# 3. Accuracy, Precision,Recall, ROC 4개의 지표로 분류모델의 정확도 평가하기

#%%

# 1. 데이터 불러오기 from TAAS: http://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS_TAS
# 해당 경로에 있는 .csv 파일명 리스트 가져오기

import os
import time
t_0= time.time()

path = '../교통사고_2021/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.xls')] ## 파일명 끝이 .xls인 경우

## csv 파일들을 DataFrame으로 불러와서 concat

death_df_org = pd.DataFrame()
for i in file_list_py:
    data = pd.read_html(path + i)
    data= data[0]
    death_df_org = pd.concat([death_df_org,data])

death_df_org = death_df_org.reset_index(drop = True)
print(f'데이터 불러오기 및 통합에 걸린 시간은 {str(time.time()-t_0)} 입니다')
death_df= death_df_org.copy()


#%%


"""
결측치 확인하기 => 피해운전자의 차종,성별,연령,상해정도를 파악하지 못하는 데이터가 6687건이 존재함
원본데이터 194094에 비하면 삭제해도 학습에 영향력을 크게 미치지 못할 뿐더러, 이러한 데이터들이
특정 편향성을 갖고 존재할 것이라 보기 어렵다(아마도 기록 누락에 의해 이러한 데이터가 존재할 것이기 때문)
그렇기 때문에 데이터를 제거해도 될것이라 판단하였음
"""
death_df.isnull().sum()
death_df.dropna(inplace=True)
death_df.isnull().sum()


#%%


#데이터 정보 찾기
print(death_df.info())
death_df.head()


# ## 종속변수의 카테고리 비율 확인하기
# - 전체 교통사고중 사망사고의 비율은 약 1.15%로, 낮은 비율을 차지하고 있다

#%%


print(death_df['사고내용'].value_counts())
ratio=2149/(138186+47072+2149)
print(f'전체 사고중 사망사고의 비율은 {str(ratio)} 입니다')


# ## 컬럼 핸들링하기
# - 불필요해보이는 컬럼 제거하기
#     1. 사고번호는 각 사고에 대해 일대일 대응되는 key값으로 나중에 데이터를 결합시키는데 사용될 수 있어 보이나, 일단
#     지금 당장 사용되지 않으므로 삭제한다
#     2. 사망자수,중상자수,경상자수,부상신고자는 '피해운전자 상해정도'에 필요한 정보가 담겨있으므로 삭제한다
#     3. 위치정보는 사용하지 않으므로 시군구도 삭제한다
# - 종속변수 '사고내용'에서 사망자를 1, 중상,경상을 0으로 하는 새로운 종속변수열을 만들기
# - 발생년월시에서 시간만 떼어내고, 정수형으로 데이터 변환하고, 22~06시를 야간, 그외 시간을 주간으로 분류하는 데이터 프레임 열 만들기
# - 피해운전자, 가해운전자 연령에서 '세' 단위 떼어내고 정수형으로 데이터 변환하기

#%%


# 경상,중상사고는 0, 사망사고는 1로 라벨링한다
index1= death_df['사고내용']=='사망사고'
death_df['사망사고여부']=index1
death_df


#%%


#불필요해보이는 컬럼 제거
death_df.drop(['사고번호','시군구','사망자수','중상자수',
               '경상자수','부상신고자수','노면상태','가해운전자 상해정도','피해운전자 상해정도','사고내용'],axis=1,inplace=True)

##시간에 따른 시계가 교통사고의 요인중 하나일 수 있으므로 발생년월일시에서 시간만 떼어 새로운 컬럼으로 만듬
time_lst= list(death_df['사고일시'])
hour_lst=[]
for time in time_lst:
    hour_lst.append((str(time)[-3:-1]))
death_df['사고시각']= hour_lst
#사고일시 컬럼은 삭제
death_df.drop(['사고일시'],axis=1,inplace=True)

#피해운전자, 가해운전자 연령 미분류 삭제하기
idx= death_df[death_df['피해운전자 연령']=='미분류'].index
death_df.drop(idx,inplace=True)
idx= death_df[death_df['피해운전자 연령']=='98세 이상'].index
death_df.drop(idx,inplace=True)

idx2= death_df[death_df['가해운전자 연령']=='미분류'].index
death_df.drop(idx2,inplace=True)

#나이 컬럼: 숫자단위만 뽑기
suspect_lst= list(death_df['가해운전자 연령'])
suspect_old= []
for old in suspect_lst:
    suspect_old.append((old)[:-1])

death_df['가해운전자 연령']=suspect_old

victim_lst= list(death_df['피해운전자 연령'])
victim_old= []
for old in victim_lst:
    victim_old.append((old)[:-1])
death_df['피해운전자 연령']=victim_old


#%%


index2= death_df['사고시각']=='22','23','24','1','2','3','4','5','6'
death_df['야간여부']=index2[0]
death_df.drop(['사고시각'],axis=1,inplace=True)


#%%


death_df=death_df.reindex(columns=['야간여부','사고유형','법규위반','기상상태','도로형태',
                                   '가해운전자 차종','가해운전자 성별','가해운전자 연령',
                                   '피해운전자 차종','피해운전자 성별','피해운전자 연령','사망사고여부'])


#%%


#데이터 자료형 변환: 문자열-> 정수형
death_df['가해운전자 연령']=death_df['가해운전자 연령'].astype('int')
death_df['피해운전자 연령']=death_df['피해운전자 연령'].astype('int')


#%%


death_df


# ![](https://scikit-learn.org/stable/_images/multi_org_chart.png)

# ## 3.범주형 변수 인코딩하기(한번만 인코딩 해야됨 두번하면 안됨)
# - 랜덤 포레스트나 트리 기반 알고리즘의 경우 LabelEncdoer를 활용한 인코딩을 하여도 되나, 그 외의 경우 One-Hot Encoding을 하여야 한다
# - One-Hot Encoding으로 sklearn에도 모듈이 있지만 pd.get_dummies가 사용편의성이 좋은듯
# - [아 졸려](https://blog.roboflow.com/precision-and-recall/#precision-and-recall-examples)

#%%


#features=['야간여부','사고유형','법규위반','기상상태','도로형태','가해운전자 차종','피해운전자 차종','사망사고여부']
features=['야간여부','사고유형','법규위반','기상상태','도로형태',
         '가해운전자 차종','가해운전자 성별',
         '피해운전자 차종','피해운전자 성별','사망사고여부']
encoded_df_org= pd.get_dummies(data=death_df,columns=features,prefix=features)
encoded_df=encoded_df_org.copy()


#%%


#데이터 분리시키기
y=encoded_df['사망사고여부_True']
X=encoded_df.drop(['사망사고여부_True','사망사고여부_False'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2)

#X_train,y_train을 다시 학습과 검증 데이터셋으로 분리
X_tr,X_val, y_tr,y_val =train_test_split(X_train,y_train,test_size=0.3,random_state=0)


#%%


#모델1. XGBclassifier

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score

# n_estimators 는 500, random_state는 예제 수행시마다 동일 예측 결과를 위해 설정한다
xgb_clf= XGBClassifier(n_estimators=300,learning_rate=0.05)

#성능 평가 지표를 auc로, 조기 중단 파라미터를 100으로 설정하고 학습을 수행시킨다
xgb_clf.fit(X_tr,y_tr,early_stopping_rounds=100,eval_metric='auc',eval_set=[(X_tr,y_tr),(X_val,y_val)])

#ROC의 정의: ROC, X축이 FPR( FP/(FP+TN), 음성인 것들의 수에서 양성이라 잘못 예측한 것의 비),
# 축은 TNR( TP/(FN+TP), 양성 중에서 양성이라 올바로 예측한 것의 비)
#ROC_AUC(Area Under Curve)는 보통 0.7이상이면 그럭저럭 괜찮고 0.8이면 좋은 모델이라 볼 수 있다

#모델 평가
roc_val= roc_auc_score(y_test,xgb_clf.predict_proba(X_test)[:,1])
recall_val= recall_score(y_test,xgb_clf.predict(X_test))
accuracy_val= accuracy_score(y_test,xgb_clf.predict(X_test))
precision_val= precision_score(y_test,xgb_clf.predict(X_test))
print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')
#recall_score가 0.036231884057971016: 대부분을 사망안함 0으로 예측안함으로 다른 지표값들을 높여서 이런 현상이 발생했을 가능성이 높음


#%%


#모델1-2. XGBclassifier - Binarizer를 적용한다

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score
from sklearn.preprocessing import Binarizer

# n_estimators 는 300, random_state는 예제 수행시마다 동일 예측 결과를 위해 설정한다
xgb_clf= XGBClassifier(n_estimators=300,learning_rate=0.05)

#성능 평가 지표를 auc로, 조기 중단 파라미터를 100으로 설정하고 학습을 수행시킨다
xgb_clf.fit(X_tr,y_tr,early_stopping_rounds=100,eval_metric='auc',eval_set=[(X_tr,y_tr),(X_val,y_val)])

#ROC의 정의: ROC, X축이 FPR( FP/(FP+TN), 음성인 것들의 수에서 양성이라 잘못 예측한 것의 비),
# 축은 TNR( TP/(FN+TP), 양성 중에서 양성이라 올바로 예측한 것의 비)
#ROC_AUC(Area Under Curve)는 보통 0.7이상이면 그럭저럭 괜찮고 0.8이면 좋은 모델이라 볼 수 있다

#Binarizer 적용하기
custom_threshold= 0.01
pred_proba_1= xgb_clf.predict_proba(X_test)[:,1].reshape(-1,1)
binarizer= Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict= binarizer.transform(pred_proba_1)

#모델 평가
roc_val= roc_auc_score(y_test,xgb_clf.predict_proba(X_test)[:,1])
recall_val= recall_score(y_test,custom_predict)
accuracy_val= accuracy_score(y_test,custom_predict)
precision_val= precision_score(y_test,custom_predict)

print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')
#recall_score가 0.036231884057971016: 대부분을 사망안함 0으로 예측안함으로 다른 지표값들을 높여서 이런 현상이 발생했을 가능성이 높음


#%%


#모델2. LightGBMClassifier
from lightgbm import LGBMClassifier

lgbm_cf= LGBMClassifier(n_estimators=400, learning_rate=0.05)
#lightGBM도 XGBoost와 마찬가지로 조기 중단 수행이 가능하다
evals=[(X_tr,y_tr),(X_val,y_val)]
lgbm_cf.fit(X_tr,y_tr,early_stopping_rounds=50, eval_metric='logloss',eval_set=evals,verbose=True)
y_pred= lgbm_cf.predict(X_test)
pred_proba= lgbm_cf.predict_proba(X_test)[:,1]

#모델 평가
roc_val= roc_auc_score(y_test,y_pred)
recall_val= recall_score(y_test,y_pred)
accuracy_val= accuracy_score(y_test,y_pred)
precision_val= precision_score(y_test,y_pred)

print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')


#%% 모델2-2. LightGBMClassifier - Binarizer 적용
from lightgbm import LGBMClassifier

lgbm_clf= LGBMClassifier(n_estimators=500, learning_rate=0.05)
#lightGBM도 XGBoost와 마찬가지로 조기 중단 수행이 가능하다
evals=[(X_tr,y_tr),(X_val,y_val)]
lgbm_clf.fit(X_tr,y_tr,early_stopping_rounds=100, eval_metric='logloss',eval_set=evals,verbose=True)
y_pred= lgbm_clf.predict(X_test)
pred_proba= lgbm_clf.predict_proba(X_test)[:,1]

#Binarizer 적용하기
custom_threshold= 0.01
y_pred_proba= lgbm_clf.predict_proba(X_test)[:,1].reshape(-1,1)
binarizer= Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict= binarizer.transform(pred_proba_1)

#모델 평가
roc_val= roc_auc_score(y_test,y_pred_proba)
recall_val= recall_score(y_test,custom_predict)
accuracy_val= accuracy_score(y_test,custom_predict)
precision_val= precision_score(y_test,custom_predict)

print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')


#%%
#모델3. 로지스틱 회귀

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr_clf= LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)
y_pred= lr_clf.predict(X_test)
y_pred_proba= lr_clf.predict_proba(X_test)[:,1]

#모델 평가
roc_val= roc_auc_score(y_test,y_pred_proba)
recall_val= recall_score(y_test,y_pred)
accuracy_val= accuracy_score(y_test,y_pred)
precision_val= precision_score(y_test,y_pred)

print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')



#%%
# 로지스틱 회귀모델3-2: Binarizer 적용
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr_clf= LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)
y_pred= lr_clf.predict(X_test)
pred_proba= lr_clf.predict_proba(X_test)[:,1]

#Binarizer 적용하기
custom_threshold= 0.01
y_pred_proba= lr_clf.predict_proba(X_test)[:,1].reshape(-1,1)
binarizer= Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict= binarizer.transform(pred_proba_1)

#모델 평가
roc_val= roc_auc_score(y_test,y_pred_proba)
recall_val= recall_score(y_test,custom_predict)
accuracy_val= accuracy_score(y_test,custom_predict)
precision_val= precision_score(y_test,custom_predict)

print('\n--------------------------------------')
print(f'ROC_AUC는 {roc_val} 입니다')
print(f'recall_score는 {recall_val} 입니다')
print(f'precision_score는 {precision_val} 입니다')
print(f'accuracy_score는 {accuracy_val} 입니다')
print('--------------------------------------')


#%%
