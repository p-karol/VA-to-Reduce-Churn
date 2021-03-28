# pacotes utilizados
pacotes = c("shiny", "shinydashboard", "shinythemes", "plotly", "shinycssloaders","tidyverse",
            "scales", "knitr", "kableExtra", "ggfortify","dplyr","plotly","FNN", "RMariaDB")

# verifica se os pacotes estão instalados, senão instala
package.check <- lapply(pacotes, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
  }
})

# conexão ao banco MySQL
dataCon <- dbConnect(MariaDB(), user = "root", password = "setPassword", dbname = "Retencao", host = "127.0.0.1", port = "3306")
# define os dados 
res <- dbSendQuery(dataCon, "SELECT md5(Dominio) as Dominio, Probabilidade, DiasAtivo, EnviosEmail,  Visitas, EspacoWeb, EspacoImap, EspacoBanco, TrafegoFtp, MediaMemoriaDiaria, MediaCpuDiariaTotal, ServidorTotalQuedas, ServidorTempoQuedas, Contatos  FROM resultado order by Probabilidade DESC limit 80000")
#res <- dbSendQuery(dataCon, "SELECT Dominio, Probabilidade, DiasAtivo, EnviosEmail,  Visitas, EspacoWeb, EspacoImap, EspacoBanco, TrafegoFtp, MediaMemoriaDiaria, MediaCpuDiariaTotal, ServidorTotalQuedas, ServidorTempoQuedas, Contatos  FROM resultado order by RAND() limit 40000")
dataDB <- dbFetch(res)
names(dataDB)[1]
