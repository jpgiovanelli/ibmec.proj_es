# install.packages(c("shiny", "randomForest"))

library(shiny)
library(randomForest)

# =====================================================
# CARREGAR DADOS
# =====================================================

dados <- read.csv(
  "C:/Users/202203377701/Downloads/GitHubBugDataSet-1.1/GitHubBugDataSet-1.1/database/junit/2014.12.04-64155f8a9babcfcf4263cf4d08253a1556e75481/junit-Class.csv",
  stringsAsFactors = TRUE
)

# =====================================================
# LIMPEZA DE COLUNAS (metadados do código)
# =====================================================

dados <- dados[, !(names(dados) %in% c(
  "ID", "Name", "LongName", "Path",
  "Line", "Column", "EndLine", "EndColumn"
))]

# =====================================================
# MODELO -> REGRESSÃO (ex: prever complexidade WMC)
# =====================================================

modelo <- randomForest(
  WMC ~ CC + LOC + CBO + RFC + LCOM5,
  data = dados,
  na.action = na.omit
)

# =====================================================
# INTERFACE
# =====================================================

ui <- fluidPage(
  
  titlePanel("Predição de Complexidade de Software (WMC)"),
  
  sidebarLayout(
    
    sidebarPanel(
      
      numericInput("cc", "Complexidade Ciclomática (CC)", 10),
      numericInput("loc", "Linhas de Código (LOC)", 200),
      numericInput("cbo", "Acoplamento (CBO)", 5),
      numericInput("rfc", "Resposta da Classe (RFC)", 10),
      numericInput("lcom", "Falta de Coesão (LCOM5)", 1),
      
      actionButton("prever", "Executar")
    ),
    
    mainPanel(
      
      h3("Previsão de WMC"),
      verbatimTextOutput("resultado")
    )
  )
)

# =====================================================
# SERVIDOR
# =====================================================

server <- function(input, output) {
  
  observeEvent(input$prever, {
    
    novo <- data.frame(
      CC = input$cc,
      LOC = input$loc,
      CBO = input$cbo,
      RFC = input$rfc,
      LCOM5 = input$lcom
    )
    
    pred <- predict(modelo, novo)
    
    output$resultado <- renderText({
      paste("WMC estimado:", round(pred, 2))
    })
  })
}

# =====================================================
# RODAR APP
# =====================================================

shinyApp(ui = ui, server = server)