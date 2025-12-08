# Projeto de Modelagem: Previsão de Preços de Casas

## Estrutura do Projeto

```
trabalho-modelagem-rain-/
├── main.ipynb                      # Notebook principal com toda análise
├── requirements.txt                # Dependências Python
├── README.md                       # Instruções do projeto
├── house-prices-advanced-regression-techniques/  # Dataset
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── ORIENTACOES_COMPLETAR_PROJETO.md  # Orientações para completar
└── exemplos_codigo.py              # Exemplos de código
```

## Dataset

**House Prices - Advanced Regression Techniques**
- **Fonte**: [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Variável-alvo (Regressão)**: `SalePrice` - Preço de venda da casa em dólares
- **Variável-alvo (Classificação)**: Criada a partir de `SalePrice` (acima/média = 1, abaixo = 0)
- **Tamanho**: 1460 observações com 80 features
- **Tipo**: Dados imobiliários de Ames, Iowa (EUA)

## Navegação do Notebook

### 1. **Introdução e Objetivos** ✅

- Contexto do projeto
- Objetivos de negócio e habilidades
- Fonte e descrição do dataset
- Licença

### 2. **Imports e Configuração**

- Bibliotecas necessárias
- Configurações visuais
- Seed para reprodutibilidade

### 3. **Carregamento e Inspeção (1️⃣)**

- Carrega CSV
- Examina shape, tipos, primeiras linhas

### 4. **Análise de Valores Ausentes (2️⃣)**

- Identifica e visualiza nulos
- Define estratégia de limpeza

### 5. **Limpeza e Tratamento (3️⃣)**

- Remove duplicatas
- Remove colunas com >50% nulos
- Preenche nulos (mediana/moda)
- Remove outliers (IQR)

### 6. **Análise Exploratória - EDA (4️⃣)**

- Estatísticas descritivas
- Distribuição da variável-alvo
- Gráficos exploratórios

### 7. **Correlações e Relações (5️⃣)**

- Matriz de correlação
- Heatmap
- Distribuições por grupo

### 8. **Preparação para Modelagem (6️⃣)**

- Codificação de variáveis categóricas
- Normalização (StandardScaler)
- Split: Train (60%) / Validation (20%) / Test (20%)

### 9. **Modelos de Regressão (7️⃣)**

- Regressão Linear Simples (statsmodels)
- Regressão Linear Múltipla (statsmodels + sklearn)
- Regressão Polinomial (sklearn)
- Métricas: MAE, RMSE, R²
- Diagnósticos: normalidade, homocedasticidade, VIF

### 10. **Modelos de Classificação (8️⃣)**

- Baseline: Classe Majoritária
- Naive Bayes
- Logistic Regression
- Métricas: accuracy, precision, recall, F1, AUC-ROC, matriz de confusão

### 11. **Validação Cruzada e Otimização (9️⃣)**

- 5-Fold Cross-Validation (K-Fold para regressão, Stratified para classificação)
- Grid Search e Random Search (sklearn)
- PyCaret: compare_models, tune_model
- Comparação antes/depois tuning

### 12. **Testes Estatísticos (🔟)**

- t-test
- ANOVA
- Qui-quadrado

### 13. **Avaliação Final no Test Set (1️⃣1️⃣)**

- Métricas finais (regressão e classificação)
- Matriz de confusão
- Curva ROC
- Classification Report
- Diagnósticos de resíduos

### 14. **Conclusões e Discussão**

- Resumo de resultados
- Insights principais
- Limitações e vieses
- Trade-offs de decisão

### 15. **Próximos Passos**

- Melhorias futuras
- Experimentos adicionais

### 16. **Referências**

- Fontes de dados
- Documentação
- Artigos relevantes

## Status do Projeto

✅ **Concluído:**
- Introdução e Objetivos
- Imports e configuração
- Carregamento de dados
- Análise de valores ausentes

⚠️ **Em correção:**
- Limpeza e tratamento de dados
- EDA (precisa adaptar para SalePrice)
- Correlações (precisa adaptar para SalePrice)
- Preparação para modelagem

❌ **A implementar:**
- Modelos de regressão
- Modelos de classificação
- Diagnósticos estatísticos
- Otimização com validação cruzada
- Testes estatísticos
- Conclusões e referências

## Como Completar

Consulte o arquivo `ORIENTACOES_COMPLETAR_PROJETO.md` para instruções detalhadas sobre como corrigir e completar cada seção do notebook.

---

## Métricas de Avaliação

### Regressão

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coeficiente de Determinação)

### Classificação (Usado neste projeto)

- **Accuracy**: Proporção de previsões corretas
- **Precision**: Proporção de previsões positivas corretas
- **Recall**: Proporção de casos positivos identificados
- **F1-Score**: Média harmônica de Precision e Recall
- **AUC-ROC**: Área sob a curva ROC
- **Confusion Matrix**: Visualização de erros

---

## Reprodutibilidade

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar notebook
jupyter notebook main.ipynb

# 3. Executar células na ordem (Kernel > Restart & Run All)
```

**Seed configurada:** `np.random.seed(42)`

---

## Notas Importantes

- Dataset deve estar no mesmo diretório que o notebook
- Todas as células possuem nomes descritivos com emojis
- Gráficos são gerados automaticamente durante execução
- Métricas são impressas após cada modelo
- Comparações tabulares ajudam na interpretação
