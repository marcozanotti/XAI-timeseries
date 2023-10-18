## test


# Installs & Load ---------------------------------------------------------

install.packages("h2o")
install.packages("DALEX")
install.packages("DALEXtra")
install.packages("lime")
install.packages("modelStudio")


library(h2o)
library(DALEX)
library(DALEXtra)
library(lime)
library(modelStudio)



# Data --------------------------------------------------------------------

# https://ema.drwhy.ai/preface.html
# https://htmlpreview.github.io/?https://github.com/ModelOriented/DALEX-docs/blob/master/vignettes/DALEX_h2o_automl.html
data(dragons)
head(dragons)



# H2O ---------------------------------------------------------------------

Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

# h2o.no_progress() 

target <- "year_of_birth"
data_h2o <- as.h2o(dragons)

model_h2o_automl <- h2o.automl(
	y = target, 
	training_frame = data_h2o, 
	max_runtime_secs = 90, 
	max_models = 20
)

leader_board <- h2o.get_leaderboard(model_h2o_automl)
head(leader_board)



# XAI ---------------------------------------------------------------------

explainer_h2o_automl <- explain_h2o(
	model = model_h2o_automl, 
	data = dragons_test[, 2:8],
	y = dragons_test$year_of_birth,
	label = "h2o automl",
	type = "regression",
	colorize = FALSE
)


# * Model Performance -----------------------------------------------------
mp_h2o_automl <- model_performance(explainer_h2o_automl)
mp_h2o_automl
plot(mp_h2o_automl)
plot(mp_h2o_automl, geom = "boxplot")


# * Feature Importance ----------------------------------------------------
fe_h2o_automl <- model_parts(explainer_h2o_automl)
plot(fe_h2o_automl)


# * Variable Response -----------------------------------------------------
# There are two main types of plots: Partial Dependence plot and Accumulated 
# Local Effects plot designed for the sake of exploring the relation between a 
# variable with the model outcome (in our case: year_of_birth)

# Partial Dependence
pdp_h2o_automl <- model_profile(
	explainer_h2o_automl, 
	variable = "weight", 
	type = "partial"
)
plot(pdp_h2o_automl)

# Accumulated Local Effect
ale_h2o_automl <- model_profile(
	explainer_h2o_automl,
	variable = "weight", 
	type = "accumulated"
)
plot(ale_h2o_automl)


# * Prediction Understanding ----------------------------------------------
new_date_birth <- dragons_test[1, ]

# Break Down
# https://medium.com/responsibleml/basic-xai-with-dalex-part-4-break-down-method-2cd4de43abdd
pb_h2o_automl <- predict_parts(
	explainer_h2o_automl,
	new_observation = new_date_birth,
	type = "break_down"
)
plot(pb_h2o_automl)

# Shap
# https://medium.com/responsibleml/basic-xai-with-dalex-part-5-shapley-values-85ceb4b58c99
shap_h2o_automl <- predict_parts(
	explainer_h2o_automl,
	new_observation = new_date_birth,
	type = "shap"
)
plot(shap_h2o_automl)

# Lime
# https://medium.com/responsibleml/basic-xai-with-dalex-part-6-lime-method-f6aab0af058a
# explainer_lime_h2o_automl <- lime(data_h2o, model_h2o_automl)
# mod_lime <- explain(new_date_birth, explainer_lime_h2o_automl, n_features = 8)
# plot_features(mod_lime)
# plot_explanations(mod_lime)

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_h2o_automl <- predict_surrogate(
	explainer = explainer_h2o_automl, 
	new_observation = new_date_birth, 
	n_features = 8, 
	n_permutations = 1000,
	type = "lime"
)
plot(lime_h2o_automl)



# * Dashboard -------------------------------------------------------------
modelStudio(explainer_h2o_automl)


