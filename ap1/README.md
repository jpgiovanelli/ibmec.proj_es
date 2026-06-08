
# 📊 AP1 - Projeto de Engenharia de Software  
## Análise Estatística de Dados de Código-Fonte de Frameworks Java

**Modalidade:** Trabalho em duplas  
**Prazo de entrega:** **14/04/2026**  
**Formato:** Artigo científico em **LaTeX (Overleaf)**  
**Template:** Modelo oficial de artigos da **SBC (Sociedade Brasileira de Computação)**  

---

# 🎯 Objetivo do Trabalho

Este trabalho tem como objetivo aplicar técnicas de **estatística descritiva e inferencial** em um dataset contendo **métricas de código-fonte de frameworks Java**, permitindo analisar características estruturais do software, identificar padrões, avaliar relações entre métricas e construir modelos estatísticos.

A proposta integra conhecimentos de **Engenharia de Software** e **Estatística**, estimulando a análise quantitativa aplicada a projetos reais de software.

---

# 📂 Fonte dos Datasets

Os datasets utilizados neste trabalho estão disponíveis em:

🔗 **GitHub Bug Dataset — versão 1.1**  
https://www.inf.u-szeged.hu/~ferenc/papers/GitHubBugDataSet/

---

# 📌 Escolha dos Datasets

Cada dupla deverá:

## 1️⃣ Escolher um framework Java diferente

Exemplos de frameworks disponíveis:

- orientdb  
- oryx  
- titan  
- junit  
- neo4j  
- antlr  
- elasticsearch  

(outros frameworks podem estar disponíveis)

---

## 2️⃣ Selecionar o dataset mais recente

Cada diretório de framework contém:

- **Dois datasets do mesmo projeto**
- Correspondentes a **versões diferentes**

📌 **Regra obrigatória:**

👉 Utilizar **o dataset mais recente** disponível para o framework escolhido.

---

# 📊 Atividades a serem realizadas

Cada dupla deverá realizar uma **análise estatística completa**, contemplando os seguintes itens:

---

# 1️⃣ Medidas de Tendência Central  

Calcular:

- Média  
- Mediana  
- Moda (quando aplicável)

Aplicar às principais variáveis numéricas do dataset.

Interpretar os resultados obtidos.

---

# 2️⃣ Medidas de Dispersão  

Calcular:

- Amplitude  
- Variância  
- Desvio padrão  

Interpretar a variabilidade das métricas.

---

# 3️⃣ Medidas de Posição Relativa  

Calcular:

- Quartis  
- Percentis  

Construir:

- Boxplots

Interpretar a distribuição relativa dos dados.

---

# 4️⃣ Construção de Gráficos  

Construir gráficos adequados aos dados, como:

- Histogramas  
- Boxplots  
- Gráficos de dispersão  
- Gráficos de barras  
- Outros gráficos que forem pertinentes 

Cada gráfico deve ser acompanhado de interpretação.

---

# 5️⃣ Avaliação de Outliers  

Identificar valores discrepantes utilizando:

- Boxplots  

Interpretar possíveis causas para a ocorrência de outliers.

---

# 6️⃣ Testes de Normalidade  

Aplicar testes estatísticos para verificar normalidade.

Sugestões:

- Shapiro-Wilk  
- Kolmogorov-Smirnov  

Interpretar os resultados obtidos.

---

# 7️⃣ Coeficientes de Correlação  

Calcular coeficientes de correlação entre variáveis numéricas utilizando o pacote:

**PerformanceAnalytics**

Função sugerida:

```r
chart.Correlation()
```

Interpretar:

- Relações fortes  
- Relações moderadas  
- Relações fracas  

---

# 8️⃣ Modelagem Estatística  

Ajustar **um modelo estatístico**, podendo ser:

## Regressão Linear  

ou  

## Regressão Logística  

A escolha dependerá das variáveis disponíveis.

A análise deve incluir:

- Ajuste do modelo  
- Interpretação dos coeficientes  
- Avaliação do desempenho  

---

# 9️⃣ Análise e Discussão dos Resultados  

Cada dupla deverá discutir:

- Padrões encontrados  
- Relações entre métricas  
- Possíveis explicações técnicas  
- Limitações da análise  

Exemplos de reflexões:

- Métricas maiores estão associadas a mais defeitos?  
- Complexidade está relacionada ao número de bugs?  
- Existem métricas fortemente correlacionadas?  

---

# 📄 Formato de Entrega

O trabalho deverá ser entregue como:

**Artigo científico**

Com as seguintes características:

- Entre **8 e 12 páginas**
- Escrito em **LaTeX**
- Desenvolvido na plataforma **Overleaf**
- Utilizando o **template oficial da SBC**

O artigo deverá conter:

- Título  
- Autores  
- Resumo  
- Introdução  
- Metodologia  
- Resultados e Discussão  
- Conclusão  
- Referências  

---

# 🧰 Repositório GitHub (Obrigatório)

Todo o material produzido deverá ser disponibilizado em:

**Um repositório no GitHub**

O repositório deverá conter:

- Código completo em **R**
- Scripts utilizados  
- Dataset selecionado  
- Figuras geradas  
- Arquivos auxiliares  

📌 **Obrigatório:**

👉 O link do repositório GitHub deve estar **citado no texto do artigo**.

---

# 📊 Critérios de Avaliação

| Critério | Pontos |
|----------|--------|
| Medidas descritivas corretas | 1,5 |
| Gráficos e visualizações | 1,5 |
| Identificação de outliers | 1,0 |
| Testes de normalidade | 1,0 |
| Análise de correlação | 1,5 |
| Modelagem estatística | 1,5 |
| Interpretação dos resultados | 1,5 |
| Organização do artigo (LaTeX/SBC) | 0,5 |

**Total: 10,0 pontos**

---

# 🧪 Ferramentas Recomendadas

- R  
- RStudio  
- Overleaf  
- GitHub  

---

# 📅 Prazo Final

📅 **Entrega até:**  
## **14 de abril de 2026**

A entrega será feita diretamente na plataforma Overleaf.

---