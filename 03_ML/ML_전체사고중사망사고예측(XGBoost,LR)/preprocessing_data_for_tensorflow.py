# 데이터 핸들링 라이브러리
import pandas as pd
import numpy as np
#데이터 시각화 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({'font.family':'AppleGothic'})
mpl.rc('axes', unicode_minus=False)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#머신러닝 성능 지표 관련 라이브러리
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score
from sklearn.metrics import precision_recall_curve

#%%

# 1. 데이터 불러오기 from TAAS: http://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS_TAS
# (이 ipynb 전용)
# 해당 경로에 있는 .csv 파일명 리스트 가져오기
def f1_read():
    import os
    import time
    t_0 = time.time()

    ## 2020년도 csv 파일들을 DataFrame으로 불러와서 concat
    path = './교통사고_2020/'
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.xls')]  ## 파일명 끝이 .xls인 경우
    death_df_2020 = pd.DataFrame()

    for i in file_list_py:
        data = pd.read_html(path + i)
        data = data[0]
        death_df_2020 = pd.concat([death_df_2020, data])

    death_df_2020 = death_df_2020.reset_index(drop=True)

    ## 2021년도 csv 파일들을 DataFrame으로 불러와서 concat
    path = './교통사고_2021/'
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.xls')]
    death_df_2021 = pd.DataFrame()

    for i in file_list_py:
        data = pd.read_html(path + i)
        data = data[0]
        death_df_2021 = pd.concat([death_df_2021, data])

    death_df_2021 = death_df_2021.reset_index(drop=True)
    death_df = pd.concat([death_df_2020, death_df_2021])
    death_df = death_df.reset_index(drop=True)
    print(f'데이터 불러오기 및 통합에 걸린 시간은 {str(time.time() - t_0)} 입니다')
    return death_df


# %%
# 2. 결측치 확인 후 제거하기

def f2_removeNaN(df):
    # 결측치 확인하기
    print(df.isnull().sum())
    # 결측치 제거하기
    df.dropna(inplace=True)
    return df


# %%
# 3. 레이블 변수 비율 살피기

def f3_ratio(df=None, label='사고내용'):
    print(df[label].value_counts())
    counts= list(df[label].value_counts())
    ratio = counts[3]/(counts[0]+counts[1]+counts[2]+counts[3])
    print(f'전체 사고중 사망사고의 비율은 {str(ratio * 100)}% 입니다')


# %%
# 4. 데이터 전처리하기 (이 ipynb 전용)

# 사망을 1, 그외의 경우를 0으로 다중분류를 이진분류로 변환시키기

def f4_preprocess(df=None, label='사고내용'):
    # 경상,중상사고는 0, 사망사고는 1로 라벨링한다
    index1 = df['사고내용'] == '사망사고'
    df['사망사고여부'] = index1

    # 불필요해보이는 컬럼 제거
    '''
    가해운전자/피해운전자 상해정도,사망자수,중상자수,경상자수,부상신고자수,사고내용은 사망사고여부에 종합하여 정보가 담기고,
    차대차인지 차대사람인지에 대한 정보가 사고유형에 담기므로 가해운전자 차종과 피해운전자 차종에서 삭제하였음
    요일은 주말과 평일로 통합하였음
    '''
    df.drop(['사고번호', '시군구', '사망자수', '중상자수', '경상자수', '부상신고자수', '노면상태',
             '가해운전자 상해정도', '피해운전자 상해정도', '사고내용', '가해운전자 차종', '피해운전자 차종'], axis=1, inplace=True)

    ##시간에 따른 시계가 교통사고의 요인중 하나일 수 있으므로 발생년월일시에서 시간만 떼어 새로운 컬럼으로 만듬
    time_lst = list(df['사고일시'])
    hour_lst = []
    for time in time_lst:
        hour_lst.append((str(time)[-3:-1]))
    df['사고시각'] = hour_lst

    # 사고일시 컬럼은 삭제
    df.drop(['사고일시'], axis=1, inplace=True)

    # 사고시각에서 22시부터 06시까지를 야간으로, 그외 시간을 야간X로 이진분류하기
    index2 = df['사고시각'] == '22', '23', '24', '1', '2', '3', '4', '5', '6'
    df['야간여부'] = index2[0]
    df.drop(['사고시각'], axis=1, inplace=True)

    # 주말과 평일로 이진분류하기
    index3 = df['요일'] == '토요일', '일요일'
    df['주말여부'] = index3[0]
    df.drop('요일', axis=1, inplace=True)

    # 남성과 여성을 1과 0으로 이진분류하기
    index4 = df['가해운전자 성별'] == '남'
    df['가해운전자 성별- 여성0 남성1'] = index4
    df.drop('가해운전자 성별', axis=1, inplace=True)

    index5 = df['피해운전자 성별'] == '남'
    df['피해운전자 성별- 여성0 남성1'] = index5
    df.drop('피해운전자 성별', axis=1, inplace=True)

    # 사망사고여부와 야간여부,주말여부, 성별을 boolean에서 정수형으로 변환하기
    df['야간여부'] = df['야간여부'].astype(int)
    df['주말여부'] = df['야간여부'].astype(int)
    df['사망사고여부'] = df['사망사고여부'].astype(int)
    df['가해운전자 성별- 여성0 남성1'] = df['가해운전자 성별- 여성0 남성1'].astype(int)
    df['피해운전자 성별- 여성0 남성1'] = df['피해운전자 성별- 여성0 남성1'].astype(int)

    # 연령 전처리하기
    # 미분류, 98세 이상 데이터 삭제
    idx = df[df['가해운전자 연령'] == '98세 이상'].index
    df.drop(idx, inplace=True)
    idx = df[df['피해운전자 연령'] == '98세 이상'].index
    df.drop(idx, inplace=True)

    idx2 = df[df['가해운전자 연령'] == '미분류'].index
    df.drop(idx2, inplace=True)
    idx2 = df[df['피해운전자 연령'] == '미분류'].index
    df.drop(idx2, inplace=True)

    # 나이 컬럼: 숫자단위만 뽑기
    suspect_lst = list(df['가해운전자 연령'])
    suspect_old = []
    for old in suspect_lst:
        suspect_old.append((old)[:-1])

    df['가해운전자 연령'] = suspect_old

    victim_lst = list(df['피해운전자 연령'])
    victim_old = []
    for old in victim_lst:
        victim_old.append((old)[:-1])
    df['피해운전자 연령'] = victim_old

    # 데이터 자료형 변환: 문자열-> 정수형
    df['가해운전자 연령'] = df['가해운전자 연령'].astype('int')
    df['피해운전자 연령'] = df['피해운전자 연령'].astype('int')

    # 나이 단위 정규화하기
    olds = df[['피해운전자 연령', '가해운전자 연령']]
    scaler = StandardScaler()
    scaled_olds = scaler.fit_transform(olds)
    df[['피해운전자 연령', '가해운전자 연령']] = scaled_olds
    df.rename({'피해운전자 연령': '피해운전자 연령(정규화 됨)', '가해운전자 연령': '가해운전자 연령(정규화 됨)'}, axis=1, inplace=True)

    # y(종속변수)가 데이터프레임 끝에 오도록 재배치하기
    df = df.reindex(columns=['주말여부', '야간여부', '사고유형', '법규위반', '기상상태', '도로형태',
                             '가해운전자 성별- 여성0 남성1', '가해운전자 연령(정규화 됨)',
                             '피해운전자 성별- 여성0 남성1', '피해운전자 연령(정규화 됨)', '사망사고여부'])

    return df

# %%
# 5. 원-핫 인코딩하기
def f5_encdoding_OneHot(df,features=['사고유형','법규위반','기상상태','도로형태']):
    encoded_df= pd.get_dummies(data=df,columns=features,prefix=features)
    return encoded_df

#%%
#6. 이상치 제거하기

def get_outlier(df=None,Label=None,column=None, weight=1.5):
    positive=df[df[Label]==1][column]
    quantile_25= np.percentile(positive.values,25)
    quantile_75= np.percentile(positive.values,75)
    #IQR을 구하고, IQR에 1.5을 곱해 최댓값과 최솟값 지점을 구한다
    IQR = quantile_75 - quantile_25
    IQR_weight= IQR * weight
    lowest_val = quantile_25- IQR_weight
    highest_val= quantile_75- IQR_weight
    #최댓값보다 크거나 최솟값보다 작은 값을 이상치 데이터로 설정하고 DataFrame 인덱스를 반환한다
    outlier_index= positive[(positive<lowest_val)|(positive>highest_val)].index
    return outlier_index

def f6_drop_outlier(df=None,label='사망사고여부',weight=1.5):
    df_copy= df.copy()
    col_names= list(df.columns)
    for col_name in col_names:
        outlier_index= get_outlier(df=df,Label=label,column=col_name,weight=1.5)
        #outlier로서 중복된 로가 존재할수도 있으므로 try-except구문을 시행한다
        try:
            df_copy.drop(outlier_index,axis=0,inplace=True)
        except:
            pass
    return df_copy

#%%
#7. 데이터셋 쪼개기- 전체를 학습과 테스트로, 학습을 다시 학습과 검증으로 & 언더샘플링
def f7_divide_dataset_and_undersampling(df):

    #독립변수, 종속변수 나누기
    y=df['사망사고여부']
    X=df.drop(['사망사고여부'],axis=1)

    #데이터 분리시키기
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2)

    #데이터 언더샘플링(지나치게 많은 y=0 데이터 제거)
    from imblearn.under_sampling import RandomUnderSampler
    rs = RandomUnderSampler()
    X_train,y_train= rs.fit_resample(X_train,y_train)
    train_df= pd.concat([X_train,y_train])
    test_df=pd.concat([X_test,y_test])



    #X_train,y_train을 다시 학습과 검증 데이터셋으로 분리
    train_df, val_df = train_test_split(train_df, test_size=0.2)


    return train_df,test_df,val_df

#%%
#8. 머신러닝 지표 계산하기

def f8_evaluate_model(y_test,y_pred,y_pred_proba):
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

    print(f'전체 테스트셋 데이터 에서 1(사망)으로 예측한 비율은 {y_pred.sum()/len(y_pred)}입니다')


#%%
#9. Gradient Boost 알고리즘 계열 feature importance 시각화

def f9_gbm_feature_importance(clf,X_train):
    # feature_importance를 배열형태로 반환
    ft_importance_values = clf.feature_importances_
    max_val= max(ft_importance_values)
    scaled_ft_importance_values= ft_importance_values/max_val
    print(ft_importance_values)

    # 정렬과 시각화를 쉽게 하기 위해 series 전환
    ft_series = pd.Series(scaled_ft_importance_values, index = X_train.columns)
    ft_top20 = ft_series.sort_values(ascending=False)[:20]

    # 시각화
    plt.figure(figsize=(8,6))
    plt.xticks(np.arange(0,1.05,0.05) ,rotation='45')
    plt.title('Feature Importance Top 20')
    sns.barplot(x=ft_top20, y=ft_top20.index)
    plt.show()

#%%
# logistic_regression 알고리즘 계열 featrue_importance 시각화
def f9_logistic_feature_importance(clf,X_train):
    # feature_importance를 배열형태로 반환
    ft_importance_values = clf.coef_[0]
    for i in range(len(ft_importance_values)):
        ft_importance_values[i]= abs(ft_importance_values[i])
    max_val= max(abs(ft_importance_values))
    scaled_ft_importance_values= ft_importance_values/max_val

    # 정렬과 시각화를 쉽게 하기 위해 series 전환
    ft_series = pd.Series(scaled_ft_importance_values, index = X_train.columns)
    ft_top20 = ft_series.sort_values(ascending=False)[:20]

    # 시각화
    plt.figure(figsize=(8,6))
    plt.xticks(np.arange(0,1.05,0.05) ,rotation='45')
    plt.title('Feature Importance Top 20')
    sns.barplot(x=ft_top20, y=ft_top20.index)
    plt.show()


#%%
#thresold값에 대한 precision과 recall 그래프 그리기

def f10_precision_recall_curve_plot(y_test,pred_proba_c1):
    precisions, recalls, thresholds= precision_recall_curve(y_test,pred_proba_c1)
    #threshold array와 이 threshold에 따른 정밀도, 재현율을 반환한다
    precisions, recalls, thresholds= precision_recall_curve(y_test,pred_proba_c1)

    #x축을 threshold값으로, y축을 정밀도, 재현율 값으로 각각 plot을 수행한다. 정밀도는 점선으로 표기한다
    plt.figure(figsize=(8,6))
    threshold_boundary= thresholds.shape[0]
    plt.plot(thresholds,precisions[0:threshold_boundary],linestyle='--',label='precision')
    plt.plot(thresholds,recalls[0:threshold_boundary],label='recall')

    #thresholds 값 X축의 scale을 0.1 단위로 변경한다
    start, end= plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))

    #x축, y축, label과 legend, 그리고 grid를 설정한다
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall Value')
    plt.legend(loc='best')
    plt.show()







#%%

#%%
