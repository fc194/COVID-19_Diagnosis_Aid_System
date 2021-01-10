library(psych)
function(input, output, session) {
  feature_importance = read.csv('www/feature_importance_LR_L1.csv', row.names = 1)
  stdscaler = read.csv('www/stdscaler.csv', row.names = 1)
  intercept = feature_importance['intercept_',]
  weight = feature_importance[rownames(feature_importance) != 'intercept_',]
  weight_name = rownames(feature_importance)[rownames(feature_importance) != 'intercept_']
  
  weight_name = weight_name[weight!=0]
  weight = weight[weight!=0]
  
  select_feature_idx = match(weight_name, rownames(stdscaler))
  stdscaler = stdscaler[select_feature_idx,]
  
  
  observeEvent(input$diagnosis, {
    if (input$TEM <= 37){
      fever_class = 0
    }
    else if (input$TEM > 37 && input$TEM <= 38){
      fever_class = 1 # mild
    }
    else if (input$TEM > 38 && input$TEM <= 39){
      fever_class = 2 # moderate
    }
    else if (input$TEM > 39){
      fever_class = 3 # severe
    }
    
    if (input$TEM <= 37){
      fever = 0
    }
    else if (input$TEM > 37){
      fever = 1
    }
    
    x = NULL
    for (wn in weight_name){
      # print(c(wn, input[[wn]]))
      if (is.null(input[[wn]])){
        x = c(x, NA)
      } else {
        x = c(x, as.numeric(input[[wn]]))
      }
    }
    x[match("Fever_class", weight_name)] = fever_class
    x[match("Fever", weight_name)] = fever
    
    # names(x) = weight_name
    
    x_std = (x-stdscaler$mean_)/stdscaler$scale_
    print(x_std)
    y_pred = logistic( sum(x_std * weight) + intercept )

    print(y_pred)
    if (y_pred < 0.5){
      sendSweetAlert(session, title = "受测者暂时低于警戒值。\nSubject is classfied as safety group.",
                     text = paste0('如无其他症状，受测者基本可以排除被感染风险。此诊断结果仅作参考。临界值：>= 0.5，预测值：', round(y_pred,8), '.',
                                   '\tSubject is temporarily classified as safety group if no other condition appeared. The diagnosis results are used for reference only. Threshold: >= 0.5，Predicted: ', round(y_pred,8), '.'), type = 'success', btn_labels = "好的 OK", btn_colors = "#3085d6", html = FALSE,
                     closeOnClickOutside = TRUE, showCloseButton = T, width = 800)
    }
    if (y_pred >= 0.5){
      sendSweetAlert(session, title = "受测者高于警戒值，建议进一步排查。\nSubject should take a further investigation.",
                     text = paste0('不必担心，在我们的模型里，判定高于警戒值的人并不代表会被最终确诊，只有真正不到20%的人才最终确诊。但我们需要进一步对您进行观察以便最终排除。临界值：>= 0.5，预测值：', round(y_pred,8), '.',
                                   '\tIn our model, subject who classified above the threshold doesn\'t directly concluded as infected. In fact, less than 20% of subjects in this group were finally identifed as infected. However, further investigations should be considered now. Threshold: >= 0.5，Predicted: ', round(y_pred,8), '.'), type = 'warning', btn_labels = "好的 OK", btn_colors = "#3085d6", html = FALSE,
                     closeOnClickOutside = TRUE, showCloseButton = T, width = 800)
    }
      
  })
  
}
