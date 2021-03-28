library(shiny)
library(crosstalk)
library(lineupjs)
library(d3scatter)
library(readr)
library(RMariaDB)

# Define UI for application that draws a histogram
ui <- fluidPage(
    titlePanel("Dashboard Retenção"),
    
    fluidRow(
       # column(12, d3scatterOutput("scatter1")),
        column(12, lineupOutput("lineup1"))
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    dataCon <- dbConnect(MariaDB(), user = "root", password = "setPassword", dbname = "Retencao", host = "127.0.0.1", port = "3306")
    res <- dbSendQuery(dataCon, "SELECT md5(Dominio) as Website, Probabilidade, DiasAtivo, EnviosEmail,  Visitas, EspacoWeb, EspacoImap, EspacoBanco, TrafegoFtp, MediaMemoriaDiaria, MediaCpuDiariaTotal, ServidorTotalQuedas, ServidorTempoQuedas, Contatos, NPS, resultado  FROM resultado order by rand()")
    dataDB <- dbFetch(res)
    #dataDB <- dbReadTable(dataDB,"resultado")
    #data_csv <- read_csv('../data/resultado1.txt')
    #shared_iris <- SharedData$new(iris)
    shared_data <- SharedData$new(dataDB)
    dbDisconnect(dataCon)
    
  #  output$scatter1 <- renderD3scatter({
    #    d3scatter(shared_data, ~Visitas, ~Probabilidade, ~Website, width = "100%")
   # })

    output$lineup1 <- renderLineup({
        #lineup(shared_data, width = "100%")
        taggle(shared_data,
               ranking=lineupRanking(sortBy=c('Probabilidade:desc')), 
           options=c(overviewMode = TRUE, sidePanel = FALSE, groupHeight = 5, groupPadding = 1, rowHeight = 15, rowPadding = 1)
             )
       # taggle(shared_data)
        #groupBy=c('resultado')
    })
}

# Run the application
shinyApp(ui = ui, server = server)
