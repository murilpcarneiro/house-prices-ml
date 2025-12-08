# Orientações para Completar o Projeto

## ✅ O que já foi feito:

1. **Introdução e Objetivos** - Adicionada seção markdown no início do notebook
2. **Imports atualizados** - Incluídas todas as bibliotecas necessárias (statsmodels, scipy, pycaret, etc.)
3. **Requirements.txt** - Atualizado com scipy e pycaret

## 🔧 O que precisa ser corrigido:

### 1. Célula 3 - Limpeza de Dados
**Problema**: Tenta converter coluna 'Date' que não existe no dataset de preços de casas.

**Solução**: Substituir por:
```python
## 3️⃣ LIMPEZA E TRATAMENTO DE DADOS

# Passo 1: Remover coluna Id
df = df.drop(columns=['Id'])

# Passo 2: Remover duplicatas
duplicates_before = len(df)
df = df.drop_duplicates()
print(f"Linhas removidas por duplicação: {duplicates_before - len(df)}")

# Passo 3: Remover colunas com >50% de nulos
cols_to_drop = missing_analysis[missing_analysis['Percentual (%)'] > 50]['Coluna'].tolist()
cols_to_drop = [col for col in cols_to_drop if col != 'Id']
print(f"\nColunas removidas (>50% nulos): {cols_to_drop}")
df = df.drop(columns=cols_to_drop)

# Passo 4: Tratamento de nulos
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if 'SalePrice' in numeric_cols:
    numeric_cols.remove('SalePrice')

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()
        if len(mode_value) > 0:
            df[col] = df[col].fillna(mode_value[0])
        else:
            df[col] = df[col].fillna('None')

# Passo 5: Detectar outliers em SalePrice
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)]
print(f"\nOutliers detectados em SalePrice: {len(outliers)}")
print(f"Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")

print(f"\nDimensões após limpeza: {df.shape}")
print(f"Valores faltantes restantes: {df.isnull().sum().sum()}")
```

### 2. Célula 4 - EDA
**Problema**: Referencia 'RainTomorrow' que não existe.

**Solução**: Substituir por análise de SalePrice:
```python
## 4️⃣ ANÁLISE EXPLORATÓRIA (EDA)

# Estatísticas descritivas
print("Estatísticas Descritivas:\n")
print(df.describe().T)

# Distribuição da variável-alvo (SalePrice)
print("\n\nDistribuição de SalePrice:")
print(df['SalePrice'].describe())

# Visualização da variável-alvo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
axes[0].hist(df['SalePrice'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Distribuição de SalePrice', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Preço de Venda ($)')
axes[0].set_ylabel('Frequência')

# Boxplot
axes[1].boxplot(df['SalePrice'], vert=True)
axes[1].set_title('Boxplot de SalePrice', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Preço de Venda ($)')

plt.tight_layout()
plt.show()
```

### 3. Célula 5 - Correlações
**Problema**: Referencia 'RainTomorrow_encoded'.

**Solução**: Substituir por análise de correlação com SalePrice:
```python
## 5️⃣ ANÁLISE DE CORRELAÇÕES E RELAÇÕES

# Matriz de correlação
plt.figure(figsize=(14, 10))
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlações com SalePrice
print("\nCorrelação com SalePrice (Top 15):")
correlations_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
print(correlations_with_target[1:16])  # Exclui a auto-correlação
```

### 4. Célula 6 - Preparação para Modelagem
**Problema**: Referencia 'RainTomorrow', 'Date', 'Location'.

**Solução**: Substituir por:
```python
## 6️⃣ PREPARAÇÃO PARA MODELAGEM

# Preparar features e target para REGRESSÃO
X_reg = df.drop(['SalePrice'], axis=1)
y_reg = df['SalePrice']

# Codificar variáveis categóricas
le_dict = {}
for col in X_reg.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_reg[col] = le.fit_transform(X_reg[col].astype(str))
    le_dict[col] = le

# Normalizar features numéricas
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_reg_scaled = pd.DataFrame(X_reg_scaled, columns=X_reg.columns)

# Split para REGRESSÃO: Train (60%), Validation (20%), Test (20%)
X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.4, random_state=42)
X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
    X_temp_reg, y_temp_reg, test_size=0.5, random_state=42)

print(f"Shapes após split (REGRESSÃO):")
print(f"  Train: X={X_train_reg.shape}, y={y_train_reg.shape}")
print(f"  Validation: X={X_val_reg.shape}, y={y_val_reg.shape}")
print(f"  Test: X={X_test_reg.shape}, y={y_test_reg.shape}")

# Criar variável de CLASSIFICAÇÃO (acima/média = 1, abaixo = 0)
median_price = y_reg.median()
y_class = (y_reg > median_price).astype(int)

print(f"\nVariável de Classificação criada:")
print(f"  Média de SalePrice: {y_reg.mean():.2f}")
print(f"  Mediana de SalePrice: {median_price:.2f}")
print(f"  Distribuição: {y_class.value_counts().to_dict()}")
print(f"  Proporção: {y_class.value_counts(normalize=True).to_dict()}")

# Split para CLASSIFICAÇÃO
X_train_clf, X_temp_clf, y_train_clf, y_temp_clf = train_test_split(
    X_reg_scaled, y_class, test_size=0.4, random_state=42, stratify=y_class)
X_val_clf, X_test_clf, y_val_clf, y_test_clf = train_test_split(
    X_temp_clf, y_temp_clf, test_size=0.5, random_state=42, stratify=y_temp_clf)

print(f"\nShapes após split (CLASSIFICAÇÃO):")
print(f"  Train: X={X_train_clf.shape}, y={y_train_clf.shape}")
print(f"  Validation: X={X_val_clf.shape}, y={y_val_clf.shape}")
print(f"  Test: X={X_test_clf.shape}, y={y_test_clf.shape}")
```

## 📋 Próximas etapas necessárias:

### 7. Modelos de Regressão
- Regressão Linear Simples (statsmodels + sklearn)
- Regressão Linear Múltipla (statsmodels + sklearn)
- Regressão Polinomial (sklearn)
- Métricas: MAE, RMSE, R²
- Diagnósticos: normalidade, homocedasticidade, VIF

### 8. Modelos de Classificação
- Naive Bayes
- Regressão Logística
- Métricas: accuracy, precision, recall, F1, AUC-ROC, matriz de confusão

### 9. Otimização
- Validação cruzada
- PyCaret: compare_models, tune_model
- Sklearn: GridSearchCV, RandomizedSearchCV

### 10. Testes Estatísticos
- t-test
- ANOVA
- Qui-quadrado

### 11. Conclusões
- Resumo de resultados
- Limitações e vieses
- Trade-offs
- Referências

## 🎯 Estrutura Final Esperada:

1. Introdução e Objetivos ✅
2. Imports ✅
3. Carregamento de Dados ✅
4. Análise de Valores Ausentes ✅
5. Limpeza e Tratamento ⚠️ (precisa correção)
6. EDA ⚠️ (precisa correção)
7. Correlações ⚠️ (precisa correção)
8. Preparação para Modelagem ⚠️ (precisa correção)
9. Modelos de Regressão ❌ (a implementar)
10. Modelos de Classificação ❌ (a implementar)
11. Avaliação e Diagnósticos ❌ (a implementar)
12. Otimização ❌ (a implementar)
13. Conclusões ❌ (a implementar)

