library(shiny)
library(shinyWidgets)
library(shinyBS)

fluidPage(
    # Math equations
    tags$head(
      tags$script(HTML("(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
                                                        m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
                                                        })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
                                                        
                                                        ga('create', 'UA-113406500-7', 'auto');
                                                        ga('send', 'pageview');"))
      ),
      tags$link(rel="stylesheet", 
                href="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.css", 
                integrity="sha384-dbVIfZGuN1Yq7/1Ocstc1lUEm+AT+/rCkibIcC/OmWo5f0EA48Vf8CytHzGrSwbQ",
                crossorigin="anonymous"),
      HTML('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.js" integrity="sha384-2BKqo+exmr9su6dir+qCw08N2ZKRucY4PrGQPPWU1A7FtlCGjmEGFqXCv5nyM5Ij" crossorigin="anonymous"></script>'),
      HTML('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous"></script>'),
      HTML('
      <script>
        document.addEventListener("DOMContentLoaded", function(){
          renderMathInElement(document.body, {
            delimiters: [{left: "$", right: "$", display: false}]
          });
        })
      </script>'),
      tags$style(HTML("
          .shiny-text-output {
            background-color:#fff;
          }
        "),
     tags$head(tags$script(HTML("document.title = 'COVID-19 Diagnosis Aid System';"))) # rename the title by JS
   ),
    
    h2("新型冠状病毒疑似感染辅助鉴别系统", br(),
       span("Suspected COVID-19 pneumonia Diagnosis Aid System", style = "font-weight: 100"),
       style = "font-family: 'Source Sans Pro';
        color: #fff; text-align: center;
        background-image: url('texturebg.png');
        padding: 50px; margin: 0px"),
    br(),
    
    fluidRow(
        column(6, offset = 2,
               p("The diagnosis results are used for reference only.")
        )
    ),
   
    fluidRow(column(8, offset = 2,hr(),h3('基本信息 Demographics'))), br(),
   
   fluidRow(column(4, offset = 2,numericInput("Age", "年龄 (Age):", 37, min = 1, max = 150)),
            column(4, selectInput("Gender", "性别 (Gender):", c("男" = 1, "女" = 0)))),
   
   fluidRow(column(8, offset = 2,hr(),h3('生命体征 Vital signes on admission'))), br(),
   fluidRow(column(4, offset = 2, numericInput("TEM", "最高体温 (Highest temperature):", 38.9, min = 20, max = 50)),
            column(4, numericInput("HR", "心率 (Heart rate):", 105, min = 1, max = 150))),
   fluidRow(column(4, offset = 2,numericInput("DIAS_BP", "舒张压 (Diastolic blood pressure):", 99), helpText('mmHg.')),
             column(4, numericInput("SYS_BP", "收缩压 (Systolic blood pressure):", 134), helpText('mmHg.'))),
   
    fluidRow(column(8, offset = 2,hr(),h3('其它症状 Other symptoms on admission'))), br(),
    
   fluidRow(column(4, offset = 2,prettyCheckbox(inputId = "Fatigue", label = "疲劳 (Fatigue)", icon = icon("check"))),
            column(4,prettyCheckbox(inputId = "Headache", label = "头痛 (Headache)", icon = icon("check")))),
   fluidRow(column(4, offset = 2,prettyCheckbox(inputId = "Shiver", label = "寒战 (Shiver)", icon = icon("check"))),
            column(4,prettyCheckbox(inputId = "Shortness of breath", label = "喘憋 (Shortness of breath)", icon = icon("check")))),
   fluidRow(column(4, offset = 2,prettyCheckbox(inputId = "Sore throat", label = "咽喉痛 (Sore throat)", icon = icon("check")))),
   
    fluidRow(column(8, offset = 2,hr(),h3('血常规 Blood routine examination'))), br(),
   
   fluidRow(column(4, offset = 2,numericInput("PLT", "血小板计数 (PLT):", 46.0), helpText('$\\times 10^9$/L, 正常范围 normal range 100-300.')),
            column(4, numericInput("MCH", "平均血红蛋白 (MCH):", 29.5), helpText('pg, 正常范围 normal range 27-34'))),
    fluidRow(column(4, offset = 2,numericInput("BASO#", "嗜碱性粒细胞绝对值 (BASO#):", 0), helpText('$\\times 10^9$/L, 正常范围 normal range 0-0.1.')),
             column(4, numericInput("EO#", "嗜酸性粒细胞绝对值 (EO#):", 0), helpText('$\\times 10^9$/L, 正常范围 normal range 0.05-0.3.'))),
    fluidRow(column(4, offset = 2,numericInput("MONO%", "单核细胞百分率 (MONO%):", 0.0800), helpText('正常范围 normal range 0.03-0.08.')),
             column(4, numericInput("IL-6", "白介素-6 (IL-6):", 11.63), helpText('pg/mL, 正常范围 normal range 0.0-5.9.'))),
   
    br(),
    fluidRow(column(8, offset = 2, actionButton("diagnosis", "诊断 Diagnosis Now", class = "btn-primary"))),
    
    br(),
    
    fluidRow(
        column(8, offset = 2,
               hr(),
               fluidRow(
                 column(8,
                        p("诊断结果仅供参考。 The diagnosis results are used for reference only."),
                        )
               )
        )
    ),
    
)