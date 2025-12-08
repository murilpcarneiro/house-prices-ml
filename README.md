# 🏠 Projeto de Modelagem: Previsão de Preços de Casas

## 📋 Descrição

Este projeto implementa uma análise completa de dados e modelagem preditiva utilizando o dataset **House Prices** do Kaggle. O objetivo é demonstrar habilidades em:

- **Análise Exploratória de Dados (EDA)**: Limpeza, tratamento de valores ausentes e investigação de relações
- **Modelagem de Regressão**: Linear simples, múltipla e polinomial
- **Modelagem de Classificação**: Naive Bayes e Regressão Logística
- **Avaliação de Desempenho**: Métricas apropriadas e diagnósticos
- **Otimização de Modelos**: Validação cruzada e tuning de hiperparâmetros

## 📊 Dataset

**House Prices - Advanced Regression Techniques**

| Propriedade                       | Descrição                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Fonte**                         | [Kaggle - House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) |
| **Tamanho**                       | 1460 observações × 80 features                                                                                        |
| **Localização**                   | Ames, Iowa (EUA)                                                                                                      |
| **Variável-alvo (Regressão)**     | `SalePrice` - Preço de venda em dólares                                                                               |
| **Variável-alvo (Classificação)** | Binária (acima/abaixo da mediana)                                                                                     |
| **Licença**                       | CC0 - Domínio Público (uso livre para fins educacionais)                                                              |

## 🔬 Metodologia

### 1. Análise Exploratória (EDA)

- Inspeção de tipos de dados e dimensionalidade
- Análise de valores ausentes e tratamento contextual
- Detecção e tratamento de outliers (IQR)
- Testes estatísticos: Shapiro-Wilk (normalidade), Breusch-Pagan (homocedasticidade)
- Visualizações: histogramas, boxplots, heatmap de correlação, pairplots, Q-Q plots

### 2. Preparação de Dados

- Codificação de variáveis categóricas (LabelEncoder)
- Normalização (StandardScaler)
- Split treino/validação/teste (60/20/20)

### 3. Modelagem

**Regressão** (predição de `SalePrice`):

- Regressão Linear Simples
- Regressão Linear Múltipla
- Regressão Polinomial (grau 2)

**Classificação** (categorização acima/abaixo da mediana):

- Gaussian Naive Bayes
- Logistic Regression

### 4. Avaliação

**Métricas de Regressão**: MAE, RMSE, R²
**Métricas de Classificação**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
**Diagnósticos**: Matriz de confusão, Curva ROC, VIF, Resíduos vs Preditos

### 5. Otimização

- Validação Cruzada (5-Fold, estratificada para classificação)
- GridSearchCV para Logistic Regression
- RandomizedSearchCV para ElasticNet
- PyCaret para comparação automática de modelos

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- pip ou conda

### Instalação

```bash
# Clone o repositório (ou extraia os arquivos)
cd trabalho-modelagem-rain-

# Instale as dependências
pip install -r requirements.txt
```

### Executar o Notebook

```bash
# Inicie o Jupyter
jupyter notebook main.ipynb

# Ou use o VS Code
code main.ipynb
```

## 📁 Estrutura do Repositório

```
trabalho-modelagem-rain-/
├── main.ipynb                                   # Notebook principal (20 células)
│   ├── Introdução e Objetivos
│   ├── EDA (8 células: limpeza, visualizações, testes)
│   ├── Modelagem de Regressão (4 células)
│   ├── Modelagem de Classificação (2 células)
│   ├── Validação Cruzada
│   ├── Otimização (GridSearch, RandomSearch, PyCaret)
│   └── Avaliação Final e Conclusões
├── requirements.txt                            # Dependências Python
├── README.md                                   # Este arquivo
├── LICENSE                                     # MIT License
├── .gitignore                                  # Standard Python
└── house-prices-advanced-regression-techniques/
    ├── train.csv                               # Dataset (1460 × 80)
    └── data_description.txt                    # Descrição das features
```

## 📦 Dependências

| Pacote           | Versão  | Uso                     |
| ---------------- | ------- | ----------------------- |
| **pandas**       | ≥2.3.0  | Manipulação de dados    |
| **numpy**        | ≥2.0.0  | Computação numérica     |
| **scipy**        | ≥1.16.0 | Testes estatísticos     |
| **matplotlib**   | ≥3.10.0 | Visualizações           |
| **seaborn**      | ≥0.13.0 | Gráficos estatísticos   |
| **scikit-learn** | ≥1.7.0  | Machine learning        |
| **statsmodels**  | ≥0.14.0 | Estatística e regressão |
| **jupyter**      | ≥1.0.0  | Notebooks interativos   |

_Opcional: `pycaret` para AutoML avançado (requer pandas<2.2)_

## 📈 Resultados Principais

### Regressão

- **Melhor Modelo**: Polynomial Regression (grau 2)
- **R² Score**: 0.850
- **RMSE**: ~$27,100

### Classificação

- **Melhor Modelo**: Logistic Regression (tuned)
- **Accuracy**: 93.5%
- **AUC-ROC**: 0.979

## 🔍 Insights Principais

1. **Features mais correlacionadas com preço**: OverallQual (0.791), GrLivArea (0.709), TotalBsmtSF (0.614)
2. **Distribuição**: Preços seguem distribuição log-normal (não-normal por Shapiro-Wilk)
3. **Limpeza**: 61 duplicatas removidas, colunas com >50% nulos descartadas
4. **Validação**: Desvio padrão baixo em CV (0.032) indica boa generalização

## ⚠️ Limitações e Considerações

- **Multicolinearidade**: VIF máximo de 5.03 (aceitável, não prejudica previsões)
- **Normalidade**: Resíduos não perfeitamente normais (afeta intervalos de confiança, não previsões)
- **Homocedasticidade**: Leve heterocedasticidade em preços extremos
- **Escopo Geográfico**: Limitado a Ames, Iowa - não generaliza para outras regiões

## 🔗 Referências

- **Dataset**: [Kaggle - House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
- **Documentação**:
  - [scikit-learn](https://scikit-learn.org/)
  - [statsmodels](https://www.statsmodels.org/)
  - [pandas](https://pandas.pydata.org/)
  - [PyCaret](https://pycaret.org/)

## 📄 Licença

Este projeto é disponibilizado sob a licença [MIT](LICENSE).

## 👤 Autor

Desenvolvido como projeto de modelagem para análise de dados e machine learning.

---

**Data**: Dezembro 2025
**Status**: ✅ Completo e Pronto para Uso
