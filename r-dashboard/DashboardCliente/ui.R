navbarPage("Customer Dashboard",
           tabPanel("Ficha do cliente",fluidPage(theme = shinytheme("flatly")),
tags$head(
  tags$style(HTML(".shiny-output-error-validation{color: red;}"))),
pageWithSidebar(
  headerPanel('Apply filters'),
  sidebarPanel(width = 4,
    selectInput('domain', 'Choose a domain:',dataDB$Dominio),
   submitButton("Update filters")
  ),
  mainPanel(
    column(8, plotlyOutput("plot1", width = 800, height=700),
           p("Texto",
             style = "font-size:25px")
    
  )
  )
))
)




