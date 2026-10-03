install.packages("readxl")
install.packages("openxlsx")
install.packages("irrCAC")
install.packages("psych")

library(readxl)
library(openxlsx)
library(irrCAC)
library(psych)

annotations <- read_excel(file.choose())

names(annotations)


death_anxiety <- data.frame(
  coder_A = annotations$Death_Anxiety_A,
  coder_B = annotations$Death_Anxiety_B
)

death_acceptance <- data.frame(
  coder_A = annotations$Death_Acceptance_A,
  coder_B = annotations$Death_Acceptance_B
)

loneliness <- data.frame(
  coder_A = annotations$Loneliness_A,
  coder_B = annotations$Loneliness_B
)

solitude <- data.frame(
  coder_A = annotations$Solitude_A,
  coder_B = annotations$Solitude_B
)

identity_confusion <- data.frame(
  coder_A = annotations$Identity_Confusion_A,
  coder_B = annotations$Identity_Confusion_B
)

identity_synthesis <- data.frame(
  coder_A = annotations$Identity_Synthesis_A,
  coder_B = annotations$Identity_Synthesis_B
)

freedom_paralysis <- data.frame(
  coder_A = annotations$Freedom_Paralysis_A,
  coder_B = annotations$Freedom_Paralysis_B
)

freedom_responsibility <- data.frame(
  coder_A = annotations$Freedom_Responsibility_A,
  coder_B = annotations$Freedom_Responsibility_B
)

meaninglessness <- data.frame(
  coder_A = annotations$Meaninglessness_A,
  coder_B = annotations$Meaninglessness_B
)

engagement <- data.frame(
  coder_A = annotations$Engagement_A,
  coder_B = annotations$Engagement_B
)

labels <- list(
  Death_Anxiety = death_anxiety,
  Death_Acceptance = death_acceptance,
  Loneliness = loneliness,
  Solitude = solitude,
  Identity_Confusion = identity_confusion,
  Identity_Synthesis = identity_synthesis,
  Freedom_Paralysis = freedom_paralysis,
  Freedom_Responsibility = freedom_responsibility,
  Meaninglessness = meaninglessness,
  Engagement = engagement
)  

results <- data.frame()

for (label_name in names(labels)) {
  
  l <- labels[[label_name]]
  
  # Calculate percentage agreement
  percentage_agreement <- mean(l$coder_A == l$coder_B) * 100
  
  # Calculate positive agreement
  pos_A <- sum(l$coder_A == 1)
  pos_B <- sum(l$coder_B == 1)
  both_pos <- sum(l$coder_A == 1 & l$coder_B == 1)
  
  if ((pos_A + pos_B) == 0) {
    pos_agreement <- NA
  } else {
    pos_agreement <- (2 * both_pos) / (pos_A + pos_B)
  }
  
  # Calculate Cohen's kappa for each label
  kappa_result <- psych::cohen.kappa(
    x = l,
    w = NULL,
    alpha = 0.05,
    levels = c(0, 1)
  )
  
  kappa_val <- kappa_result$kappa[1]
  
  # Select 95% confidence interval for unweighted kappa
  kappa_ci_lower <- kappa_result$confid[1, "lower"]
  kappa_ci_upper <- kappa_result$confid[1, "upper"]
  
  # Calculate Gwet's AC1 for each label
  ac1_result <- gwet.ac1.raw(ratings = l,
                             weights = "unweighted",
                             categ.labels = c(0, 1),
                             conflev = 0.95,
                             N = Inf
  )
  
  ac1_value <- ac1_result$est$coeff.val
  ac1_se <- ac1_result$est$coeff.se
  
  # Establish benchmarks from Landis and Koch and calculate probabilities for CIMP
  benchmark <- irrCAC::landis.koch.bf(
    coeff = ac1_value,
    se = ac1_se
  )
  
  cat("\n", label_name, "\n")
  print(benchmark)

  cumprob <- benchmark[ , "CumProb"]
  eligible_rows <- which(cumprob >= 0.95)
  
  if (length(eligible_rows) > 0) {
    cimp_interpretation <- benchmark[eligible_rows[1], "Landis-Koch"]
  } else {
    cimp_interpretation <- NA
  }
  
  cat("CIMP interpretation:", cimp_interpretation, "\n")
  # Add results to a dataframe
  results <- rbind(
    results,
    data.frame(
      Label = label_name,
      Positive_Ratings_A = pos_A,
      Positive_Ratings_B = pos_B,
      Positive_Agreement = pos_agreement,
      Percentage_Agreement = percentage_agreement,
      Cohen_Kappa = kappa_val,
      Kappa_CI_Lower = kappa_ci_lower,
      Kappa_CI_Upper = kappa_ci_upper,
      Gwet_AC1 = ac1_value,
      AC1_SE = ac1_se,
      AC1_Conf_Interval = ac1_result$est$conf.int,
      CIMP_Interpretation = cimp_interpretation
    )
  )
}

# View results table
print(results)

# Save results in an Excel-file
write.xlsx(results, "IRR_results.xlsx", overwrite = TRUE)