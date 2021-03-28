function(input, output, session) {
  
    selectedDataAllDomain <- reactive({
      dataDB %>%
        select(1:14)%>%
        filter(dataDB$Dominio != input$domain) 
    })
    
    selectedDataFilter <- reactive({
      selectedDataAllDomain() %>%
        filter(selectedDataAllDomain()$position %in% input$position,
               selectedData1()$foot %in% input$foot) %>%
        filter(overall >= input$overall[1]) %>%
        filter(overall <= input$overall[2]) %>%
        filter(height >= input$height[1])  %>%
        filter(height <= input$height[2]) 
    })
    
    selectedDataInputDomain <- reactive({
      dataDB %>%
        select(2:14)%>%
        filter(dataDB$Dominio == input$domain) 
    })
    
    selectedData <- reactive({
      dataDB %>%
        select(1:14) %>%
      filter(dataDB$Dominio != input$domain) 
    })
    
    selectedData2 <- reactive({
      selectedData() %>%
        select(2:14)
    })
    
    selectedDataSimilarDomains <- reactive({
      as.numeric(knnx.index(selectedData2(), selectedData2(), k=3))
    })
    
    selectedDataFinal <- reactive({
      selectedData()[selectedDataSimilarDomains(),]
    })
    
    selectedDataFinalWithoutDomain <- reactive({
      selectedDataFinal() %>%
        select(2:14)
    })

    
    # Combine the selected variables into a new data frame
  output$plot1 <- renderPlotly({
    plot_ly(
      type = 'scatterpolar',
      mode = "closest",
      fill = 'toself'
    ) %>%
      add_trace(
        r = as.matrix(selectedDataInputDomain()[1,]),
        theta = c("Probabilidade", "DiasAtivo", "EnviosEmail",  "Visitas", "EspacoWeb",
              "EspacoImap", "EspacoBanco", "TrafegoFtp", "MediaMemoriaDiaria", 
              "MediaCpuDiariaTotal", "ServidorTotalQuedas", "ServidorTempoQuedas", "Contatos"),
        showlegend = TRUE,
        mode = "markers",
        name = input$domain
        
      ) %>%
      add_trace(
        r = as.matrix(selectedDataFinalWithoutDomain()[2,]),
        theta = c("Probabilidade", "DiasAtivo", "EnviosEmail",  "Visitas", "EspacoWeb",
               "EspacoImap", "EspacoBanco", "TrafegoFtp", "MediaMemoriaDiaria", 
             "MediaCpuDiariaTotal", "ServidorTotalQuedas", "ServidorTempoQuedas", "Contatos"),
        showlegend = TRUE,
        mode = "markers",
        visible="legendonly",
        name = selectedDataFinal()[2,1]
        
      ) %>%
      add_trace(
        r = as.matrix(selectedDataFinalWithoutDomain()[3,]),
        theta = c("Probabilidade", "DiasAtivo", "EnviosEmail",  "Visitas", "EspacoWeb",
                  "EspacoImap", "EspacoBanco", "TrafegoFtp", "MediaMemoriaDiaria", 
                  "MediaCpuDiariaTotal", "ServidorTotalQuedas", "ServidorTempoQuedas", "Contatos"),
        showlegend = TRUE,
        mode = "markers",
        visible="legendonly",
        name = selectedDataFinal()[3,1]
        
      ) %>%
      layout(
        polar = list(
          radialaxis = list(
            visible = T,
            range = c(0,100)
          )
       ),
     
     showlegend=TRUE
        
      )
    
  })
  
}



