"""
Exemplos de código para implementar no notebook
Copie e adapte conforme necessário
"""

# ============================================================================
# MODELOS DE REGRESSÃO
# ============================================================================

# 1. REGRESSÃO LINEAR SIMPLES (statsmodels)
from statsmodels.formula.api import ols

# Exemplo com uma feature
model_simple = ols('SalePrice ~ GrLivArea', data=df).fit()
print(model_simple.summary())

# 2. REGRESSÃO LINEAR MÚLTIPLA (statsmodels)
# Selecionar features mais correlacionadas
top_features = correlation_matrix['SalePrice'].sort_values(ascending=False)[1:6].index.tolist()
formula = 'SalePrice ~ ' + ' + '.join(top_features)
model_multiple = ols(formula, data=df).fit()
print(model_multiple.summary())

# 3. REGRESSÃO LINEAR MÚLTIPLA (sklearn)
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_train = lr_model.predict(X_train_reg)
y_pred_val = lr_model.predict(X_val_reg)

# Métricas
mae_train = mean_absolute_error(y_train_reg, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train_reg, y_pred_train))
r2_train = r2_score(y_train_reg, y_pred_train)

mae_val = mean_absolute_error(y_val_reg, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val_reg, y_pred_val))
r2_val = r2_score(y_val_reg, y_pred_val)

print(f"Train - MAE: {mae_train:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_train:.4f}")
print(f"Val - MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.4f}")

# 4. REGRESSÃO POLINOMIAL (sklearn)
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_reg)
X_val_poly = poly_features.transform(X_val_reg)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)
y_pred_poly = poly_model.predict(X_val_poly)

# ============================================================================
# DIAGNÓSTICOS DE REGRESSÃO
# ============================================================================

# 1. Normalidade dos resíduos
residuals = y_val_reg - y_pred_val

# Teste de Shapiro-Wilk
from scipy.stats import shapiro
stat, p_value = shapiro(residuals[:5000])  # Limita a 5000 para performance
print(f"Shapiro-Wilk: stat={stat:.4f}, p-value={p_value:.4f}")

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=50, edgecolor='black')
axes[0].set_title('Distribuição dos Resíduos')
axes[0].set_xlabel('Resíduos')
axes[0].set_ylabel('Frequência')

from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot dos Resíduos')
plt.tight_layout()
plt.show()

# 2. Homocedasticidade (Teste de Breusch-Pagan)
from statsmodels.stats.diagnostic import het_breuschpagan

# Para statsmodels
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model_multiple.resid, model_multiple.model.exog)
print(f"Breusch-Pagan: LM={lm:.4f}, p-value={lm_pvalue:.4f}")

# Visualização
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_val, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')
plt.title('Resíduos vs Valores Preditos (Homocedasticidade)')
plt.show()

# 3. Multicolinearidade (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calcular VIF para features numéricas
vif_data = pd.DataFrame()
vif_data["Feature"] = top_features
vif_data["VIF"] = [variance_inflation_factor(df[top_features].values, i) 
                   for i in range(len(top_features))]
print(vif_data)

# ============================================================================
# MODELOS DE CLASSIFICAÇÃO
# ============================================================================

# 1. NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_clf, y_train_clf)
y_pred_nb = nb_model.predict(X_val_clf)
y_proba_nb = nb_model.predict_proba(X_val_clf)[:, 1]

# Métricas
acc_nb = accuracy_score(y_val_clf, y_pred_nb)
prec_nb = precision_score(y_val_clf, y_pred_nb)
rec_nb = recall_score(y_val_clf, y_pred_nb)
f1_nb = f1_score(y_val_clf, y_pred_nb)
auc_nb = roc_auc_score(y_val_clf, y_proba_nb)

print(f"Naive Bayes - Acc: {acc_nb:.4f}, Prec: {prec_nb:.4f}, Rec: {rec_nb:.4f}, F1: {f1_nb:.4f}, AUC: {auc_nb:.4f}")

# 2. REGRESSÃO LOGÍSTICA
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_clf, y_train_clf)
y_pred_lr = lr_clf.predict(X_val_clf)
y_proba_lr = lr_clf.predict_proba(X_val_clf)[:, 1]

# Métricas
acc_lr = accuracy_score(y_val_clf, y_pred_lr)
prec_lr = precision_score(y_val_clf, y_pred_lr)
rec_lr = recall_score(y_val_clf, y_pred_lr)
f1_lr = f1_score(y_val_clf, y_pred_lr)
auc_lr = roc_auc_score(y_val_clf, y_proba_lr)

print(f"Logistic Regression - Acc: {acc_lr:.4f}, Prec: {prec_lr:.4f}, Rec: {rec_lr:.4f}, F1: {f1_lr:.4f}, AUC: {auc_lr:.4f}")

# Matriz de Confusão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val_clf, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Curva ROC
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_val_clf, y_proba_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# ============================================================================
# VALIDAÇÃO CRUZADA
# ============================================================================

from sklearn.model_selection import cross_validate, StratifiedKFold

# Para Regressão
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_reg = {'mae': 'neg_mean_absolute_error', 
               'rmse': 'neg_root_mean_squared_error', 
               'r2': 'r2_score'}

cv_results_reg = cross_validate(lr_model, X_train_reg, y_train_reg, 
                                cv=cv_reg, scoring=scoring_reg)
print("CV Results (Regression):")
for metric in scoring_reg.keys():
    scores = cv_results_reg[f'test_{metric}']
    print(f"  {metric.upper()}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Para Classificação
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_clf = {'accuracy': 'accuracy', 
               'precision': 'precision', 
               'recall': 'recall', 
               'f1': 'f1', 
               'roc_auc': 'roc_auc'}

cv_results_clf = cross_validate(lr_clf, X_train_clf, y_train_clf, 
                                cv=cv_clf, scoring=scoring_clf)
print("\nCV Results (Classification):")
for metric in scoring_clf.keys():
    scores = cv_results_clf[f'test_{metric}']
    print(f"  {metric.upper()}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ============================================================================
# OTIMIZAÇÃO COM GRID SEARCH
# ============================================================================

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search para Regressão Logística
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                       param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_lr.fit(X_train_clf, y_train_clf)

print(f"Melhores parâmetros: {grid_lr.best_params_}")
print(f"Melhor F1-Score (CV): {grid_lr.best_score_:.4f}")

# Random Search (mais rápido para espaços grandes)
from scipy.stats import uniform, randint

param_dist_lr = {
    'C': uniform(0.001, 100),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

random_lr = RandomizedSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                               param_distributions=param_dist_lr, n_iter=20,
                               cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_lr.fit(X_train_clf, y_train_clf)

# ============================================================================
# PYCARET (se disponível)
# ============================================================================

if PYCARET_AVAILABLE:
    # Setup para Regressão
    reg_setup = setup(data=df, target='SalePrice', session_id=42, 
                     train_size=0.8, silent=True)
    
    # Comparar modelos
    best_reg = compare_models(sort='RMSE', n_select=3)
    
    # Tunar melhor modelo
    tuned_reg = tune_model(best_reg[0], optimize='RMSE')
    
    # Setup para Classificação
    clf_setup = setup(data=df.drop('SalePrice', axis=1).assign(HighPrice=y_class), 
                     target='HighPrice', session_id=42, 
                     train_size=0.8, silent=True)
    
    # Comparar modelos
    best_clf = compare_models(sort='F1', n_select=3)
    
    # Tunar melhor modelo
    tuned_clf = tune_model(best_clf[0], optimize='F1')

# ============================================================================
# TESTES ESTATÍSTICOS
# ============================================================================

# 1. t-test: Comparar preços entre dois grupos
from scipy.stats import ttest_ind

# Exemplo: Comparar preços por qualidade
high_qual = df[df['OverallQual'] >= 7]['SalePrice']
low_qual = df[df['OverallQual'] < 7]['SalePrice']

t_stat, p_value = ttest_ind(high_qual, low_qual)
print(f"t-test: t={t_stat:.4f}, p-value={p_value:.4f}")

# 2. ANOVA: Comparar preços entre múltiplos grupos
from scipy.stats import f_oneway

# Exemplo: Preços por número de quartos
groups = [df[df['BedroomAbvGr'] == i]['SalePrice'] for i in range(1, 6)]
f_stat, p_value = f_oneway(*groups)
print(f"ANOVA: F={f_stat:.4f}, p-value={p_value:.4f}")

# 3. Qui-quadrado: Testar associação entre variáveis categóricas
from scipy.stats import chi2_contingency

# Exemplo: Associação entre CentralAir e HighPrice
contingency_table = pd.crosstab(df['CentralAir'], y_class)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square: χ²={chi2:.4f}, p-value={p_value:.4f}, dof={dof}")

