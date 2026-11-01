import numpy as np, pandas as pd, json, pathlib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
np.random.seed(42)
root=pathlib.Path(__file__).parent; (root/"data").mkdir(exist_ok=True); (root/"results").mkdir(exist_ok=True)
n=4500
tenure=np.random.randint(1,73,size=n); monthly=np.round(np.random.normal(65,25,n).clip(20,120),2); total=np.round(monthly*tenure*np.random.uniform(.9,1.1,n),2)
contract=np.random.choice([0,1,2],size=n,p=[.55,.25,.2]); support=np.random.randint(0,2,size=n); online=np.random.randint(0,2,size=n)
logit=-1.2 -0.04*tenure +0.015*monthly -0.9*contract +0.5*support*(-1) + np.random.normal(0,0.7,n)
# month-to-month (contract 0) churns more; add effect
logit=logit + np.where(contract==0,0.6,-0.3)
prob=1/(1+np.exp(-logit)); churn=(np.random.random(n)<prob).astype(int)
df=pd.DataFrame({"tenure_months":tenure,"monthly_charges":monthly,"total_charges":total,"contract_type":contract,"tech_support":support,"online_security":online,"churn":churn})
df.to_csv(root/"data/customers.csv",index=False)
X=df.drop(columns=["churn"]); y=df.churn
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=cross_val_score(GradientBoostingClassifier(random_state=42),Xtr,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="roc_auc")
model=GradientBoostingClassifier(random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte); proba=model.predict_proba(Xte)[:,1]
summary={"rows":n,"churn_rate":round(float(y.mean()),4),"cv_roc_auc":round(float(cv.mean()),4),"test_roc_auc":round(float(roc_auc_score(yte,proba)),4),"precision":round(float(precision_score(yte,pred)),4),"recall":round(float(recall_score(yte,pred)),4),"f1":round(float(f1_score(yte,pred)),4),"feature_importances":{c:round(float(v),4) for c,v in zip(X.columns, model.feature_importances_)},"top_feature":str(X.columns[int(np.argmax(model.feature_importances_))]),"high_risk_segment":"month-to-month contract, low tenure (from EDA in README)"}
(root/"results/summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
