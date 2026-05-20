# install.packages(c("shiny", "randomForest", "ggplot2"))

library(shiny)
library(randomForest)
library(ggplot2)

# =====================================================
# CARREGAR DADOS
# =====================================================

dados <- read.csv(
  "C:/Users/202203377701/Downloads/GitHubBugDataSet-1.1/GitHubBugDataSet-1.1/database/junit/2014.12.04-64155f8a9babcfcf4263cf4d08253a1556e75481/junit-Class.csv",
  stringsAsFactors = TRUE
)

# =====================================================
# LIMPEZA
# =====================================================

dados <- dados[, !(names(dados) %in% c(
  "ID", "Name", "LongName", "Path",
  "Line", "Column", "EndLine", "EndColumn"
))]

dados <- na.omit(dados)

# =====================================================
# MODELO
# =====================================================

modelo <- randomForest(
  WMC ~ CC + LOC + CBO + RFC + LCOM5,
  data = dados
)

# Importância das variáveis
importancia <- data.frame(
  Variavel = rownames(importance(modelo)),
  Importancia = importance(modelo)[, 1]
)

# =====================================================
# UI
# =====================================================

ui <- fluidPage(
  
  titlePanel("Dashboard de Qualidade de Software"),
  
  tabsetPanel(
    
    # =========================
    # TAB 1 - PREDIÇÃO
    # =========================
    tabPanel("Previsão",
             
             sidebarLayout(
               
               sidebarPanel(
                 numericInput("cc", "CC", 10),
                 numericInput("loc", "LOC", 200),
                 numericInput("cbo", "CBO", 5),
                 numericInput("rfc", "RFC", 10),
                 numericInput("lcom", "LCOM5", 1),
                 
                 actionButton("btn", "Prever")
               ),
               
               mainPanel(
                 h3("Resultado"),
                 verbatimTextOutput("predicao")
               )
             )
    ),
    
    # =========================
    # TAB 2 - IMPORTÂNCIA
    # =========================
    tabPanel("Importância das Variáveis",
             plotOutput("plot_importancia")
    ),
    
    # =========================
    # TAB 3 - DISTRIBUIÇÃO
    # =========================
    tabPanel("Distribuição WMC",
             plotOutput("plot_dist")
    ),
    
    # =========================
    # TAB 4 - INSIGHT
    # =========================
    tabPanel("Análise",
             plotOutput("plot_scatter")
    )
  )
)

# =====================================================
# SERVER
# =====================================================

server <- function(input, output) {
  
  # -------------------------
  # PREVISÃO
  # -------------------------
  observeEvent(input$btn, {
    
    novo <- data.frame(
      CC = input$cc,
      LOC = input$loc,
      CBO = input$cbo,
      RFC = input$rfc,
      LCOM5 = input$lcom
    )
    
    pred <- predict(modelo, novo)
    
    output$predicao <- renderText({
      paste("WMC estimado:", round(pred, 2))
    })
  })
  
  # -------------------------
  # IMPORTÂNCIA
  # -------------------------
  output$plot_importancia <- renderPlot({
    
    ggplot(importancia, aes(x = reorder(Variavel, Importancia),
                            y = Importancia)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "Importância das Variáveis",
           x = "Variável",
           y = "Importância") +
      theme_minimal()
  })
  
  # -------------------------
  # DISTRIBUIÇÃO WMC
  # -------------------------
  output$plot_dist <- renderPlot({
    
    ggplot(dados, aes(x = WMC)) +
      geom_histogram(fill = "darkgreen", bins = 30) +
      labs(title = "Distribuição do WMC",
           x = "WMC",
           y = "Frequência") +
      theme_minimal()
  })
  
  # -------------------------
  # RELAÇÃO CC vs WMC
  # -------------------------
  output$plot_scatter <- renderPlot({
    
    ggplot(dados, aes(x = CC, y = WMC)) +
      geom_point(alpha = 0.5, color = "darkred") +
      geom_smooth(method = "lm", color = "blue") +
      labs(title = "Relação entre CC e WMC",
           x = "CC",
           y = "WMC") +
      theme_minimal()
  })
}

# =====================================================
# RODAR APP
# =====================================================

shinyApp(ui = ui, server = server)
