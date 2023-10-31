## Exam Script


# Installs & Load ---------------------------------------------------------

install.packages("tidyverse")
install.packages("timetk")
install.packages("tidymodels")
install.packages("modeltime")
install.packages("modeltime.h2o")
install.packages("h2o")
install.packages("DALEX")
install.packages("DALEXtra")
install.packages("lime")


library(tidyverse)
library(timetk)
library(tidymodels)
library(modeltime)
library(modeltime.h2o)
library(h2o)
library(DALEX)
library(DALEXtra)
library(lime)



# Data --------------------------------------------------------------------

# * Import ----------------------------------------------------------------
m4 <- read_csv("data/m4.csv")
m4


# * Visualization ---------------------------------------------------------

m4$period |> unique()
freq <- "Hourly"
m4_freq <- m4 |> filter(period == freq) |> select(-period)
m4_freq |> 
	plot_time_series(
		.date_var = date, .value = value, 
		.facet_vars = id, .facet_scales = "free", .facet_ncol = 2, .facet_nrow = 1,
		.smooth = FALSE, .title = "", .trelliscope = TRUE
	)

ts_id <- "H413"
m4_ts <- m4_freq |> filter(id == ts_id) |> select(-id, -type)
m4_ts |> 
	plot_time_series(
		.date_var = date, .value = value, 
		.smooth = FALSE, .title = ts_id, .interactive = FALSE
	)


# * Feature Engineering ---------------------------------------------------

horizon <- 48
lag_period <- 48
rolling_periods <- c(12, 24)

m4_ts_prep <- m4_ts |> 
	# standardization
	mutate(value = standardize_vec(value)) |>  
	# add lags
	tk_augment_lags(value, .lags = lag_period) |>
	# add rolling features
	tk_augment_slidify(
		value_lag48, mean, .period = rolling_periods, .align = "center", .partial = TRUE
	) |>
	rename_with(~ str_remove_all(., "(value_)|(lag48_)")) |> 
	rename_with(~ str_remove_all(., "_")) |> 
	# calendar features
	tk_augment_timeseries_signature(date) |> 
	select(-matches("(diff)|(iso)|(xts)|(lbl)|(year)|(half)|(quarter)|(minute)|(second)|(hour12)|(qday)|(yday)")) |> 
	mutate(
		index.num = standardize_vec(index.num),
		am.pm = as.factor(am.pm),
	) |> 
	# fourier series
	tk_augment_fourier(date, .periods = rolling_periods, .K = 2) |> 
	rename_with(~ str_remove_all(., "date_")) |> 
	# drop na
	drop_na()
glimpse(m4_ts_prep)	


# * Train / Test Sets -----------------------------------------------------

splits <- m4_ts_prep |> time_series_split(assess = horizon, cumulative = TRUE)
splits |>
	tk_time_series_cv_plan() |>
	plot_time_series_cv_plan(date, value, .interactive = FALSE)



# Modelling ---------------------------------------------------------------

# * H2O Setup -------------------------------------------------------------

Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

h2o.no_progress() 

train_h2o <- as.h2o(training(splits))
test_h2o <- as.h2o(testing(splits))
target <- "value"
x_vars <- setdiff(names(train_h2o), c(target, "date"))


# * Model Estimation ------------------------------------------------------

# Algorithms:
# - DRF (This includes both the Distributed Random Forest (DRF) and
#   Extremely Randomized Trees (XRT) models.
# - GLM (Generalized Linear Model with regularization)
# - XGBoost (XGBoost GBM)
# - GBM (H2O GBM)
# - DeepLearning (Fully-connected multi-layer artificial neural network)
# - StackedEnsemble (Stacked Ensembles, includes an ensemble of all the
#   base models and ensembles using subsets of the base models)
model_h2o_automl <- h2o.automl(
	y = target, x = x_vars,
	training_frame = train_h2o,
	max_runtime_secs = 120,
	max_runtime_secs_per_model = 20,
	max_models = 50,
	nfolds = 5,
	sort_metric = "rmse",
	verbosity = NULL,
	seed = 123
)


# * Model Performance -----------------------------------------------------

leader_board <- h2o.get_leaderboard(model_h2o_automl)
head(leader_board)

h2o_best <- h2o.get_best_model(model_h2o_automl)
h2o_best
h2o.performance(h2o_best, train_h2o)
h2o.performance(h2o_best, test_h2o)

train_pred <- as.matrix(h2o.predict(h2o_best, newdata = train_h2o))[, 1]
test_pred <- as.matrix(h2o.predict(h2o_best, newdata = test_h2o))[, 1]

m4_ts_pred <- m4_ts_prep |> 
	select(date, value) |> 
	mutate(
	  pred = c(train_pred, test_pred),
		type = c(rep("train", length(train_pred)), rep("test", length(test_pred)))
	)
m4_ts_pred |> 
	filter(type == "train") |> 
	pivot_longer(cols = c(value, pred)) |> 
	plot_time_series(
		.date_var = date, .value = value, 
		.color_var = name, .smooth = FALSE, .interactive = FALSE
	)
m4_ts_pred |> 
	filter(type == "test") |> 
	pivot_longer(cols = c(value, pred)) |> 
	plot_time_series(
		.date_var = date, .value = value, 
		.color_var = name, .smooth = FALSE, .interactive = FALSE
	)



# Modelling with modeltime.h2o --------------------------------------------

# * H2O Setup -------------------------------------------------------------

Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

h2o.no_progress() 


# * Engine ----------------------------------------------------------------

# Algorithms:
# - DRF (This includes both the Distributed Random Forest (DRF) and
#   Extremely Randomized Trees (XRT) models.
# - GLM (Generalized Linear Model with regularization)
# - XGBoost (XGBoost GBM)
# - GBM (H2O GBM)
# - DeepLearning (Fully-connected multi-layer artificial neural network)
# - StackedEnsemble (Stacked Ensembles, includes an ensemble of all the
#   base models and ensembles using subsets of the base models)

model_spec_h2o <- automl_reg(mode = "regression") |>
	set_engine(
		engine = "h2o",
		max_runtime_secs = 120,
		max_runtime_secs_per_model = 20,
		max_models = 50,
		nfolds = 5,
		sort_metric = "rmse",
		verbosity = NULL,
		seed = 123
	)
model_spec_h2o

rcp_spec <- recipe(value ~ ., data = training(splits))


# * Workflows -------------------------------------------------------------

# This step will take some time depending on your engine specifications
wrkfl_fit_h2o <- workflow() |>
	add_model(model_spec_h2o) |>
	add_recipe(rcp_spec) |>
	fit(training(splits))
wrkfl_fit_h2o

wrkfl_fit_h2o |> automl_leaderboard() |> View()

save_h2o_model(wrkfl_fit_h2o, "data/h2o_model/", overwrite = TRUE)
# wrkfl_fit_h2o <- load_h2o_model("data/h2o_model/")


# * Calibration, Evaluation & Plotting ------------------------------------

# Function to calibrate models, evaluate their accuracy and plot results
calibrate_evaluate_plot <- function(
		..., splits, actual_data, type = "testing", updated_desc = NULL
) {
	
	if (type == "testing") {
		new_data <- testing(splits)
	} else {
		new_data <- training(splits) |> drop_na()
	}
	
	calibration_tbl <- modeltime_table(...)
	
	if (!is.null(updated_desc)) {
		for (i in seq_along(updated_desc)) {
			calibration_tbl <- calibration_tbl |> 
				update_model_description(.model_id = i, .new_model_desc = updated_desc[i])
		}
	}
	
	calibration_tbl <- calibration_tbl |> 
		modeltime_calibrate(new_data)
	
	print(calibration_tbl |> modeltime_accuracy())
	
	print(
		calibration_tbl |> 
			modeltime_forecast(new_data = new_data, actual_data = actual_data) %>%
			plot_modeltime_forecast(.conf_interval_show = FALSE)
	)
	
	return(invisible(calibration_tbl))
	
}

calibrate_evaluate_plot(wrkfl_fit_h2o, splits = splits, actual_data = m4_ts_prep)



# XAI ---------------------------------------------------------------------

# * Explainer -------------------------------------------------------------

# https://dalex.drwhy.ai/
# h2o
explainer_h2o_automl <- explain_h2o(
	model = h2o_best, 
	data = select(testing(splits), -date, -value),
	y = testing(splits)$value,
	label = "h2o automl",
	type = "regression",
	colorize = FALSE
)

# modeltime.h2o
# explainer_h2o_automl <- explain_h2o(
# 	model = wrkfl_fit_h2o$fit$fit, 
# 	data = select(testing(splits), -date, -value),
# 	y = testing(splits)$value,
# 	label = "h2o automl",
# 	type = "regression",
# 	colorize = FALSE
# )


# * Global Explanation ----------------------------------------------------

# ** Performance ----------------------------------------------------------
mp_h2o_automl <- model_performance(explainer_h2o_automl)
mp_h2o_automl
plot(mp_h2o_automl)
plot(mp_h2o_automl, geom = "boxplot")

# ** Feature Importance ---------------------------------------------------
fe_h2o_automl <- model_parts(explainer_h2o_automl)
plot(fe_h2o_automl)
plot(fe_h2o_automl, max_vars = 10)

# ** Variable Response ----------------------------------------------------
# Partial Dependence
pdp_h2o_automl <- model_profile(
	explainer_h2o_automl, 
	variable = c("hour", "wday"), 
	type = "partial"
)
plot(pdp_h2o_automl)

# ** Diagnostics ----------------------------------------------------------
md_h2o_automl <- model_diagnostics(explainer_h2o_automl)
md_h2o_automl
plot(md_h2o_automl)
plot(md_h2o_automl, variable = "hour")


# * Local Explanation -----------------------------------------------------

new_value <- testing(splits) |> filter(date == ymd_hms("2017-07-23 02:00:00"))
new_value <- testing(splits) |> filter(date == ymd_hms("2017-07-24 04:00:00"))
new_value <- testing(splits) |> filter(date == ymd_hms("2017-07-23 15:00:00"))

# ** Prediction Understanding ----------------------------------------------
# Break Down
# https://medium.com/responsibleml/basic-xai-with-dalex-part-4-break-down-method-2cd4de43abdd
bd_h2o_automl <- predict_parts(
	explainer_h2o_automl,
	new_observation = new_value,
	type = "break_down"
)
plot(bd_h2o_automl)

# Shap
# https://medium.com/responsibleml/basic-xai-with-dalex-part-5-shapley-values-85ceb4b58c99
shap_h2o_automl <- predict_parts(
	explainer_h2o_automl,
	new_observation = new_value,
	type = "shap"
)
plot(shap_h2o_automl)

# Lime
# https://medium.com/responsibleml/basic-xai-with-dalex-part-6-lime-method-f6aab0af058a
# explainer_lime_h2o_automl <- lime(train_h2o, h2o_best)
# mod_lime <- explain(new_value, explainer_lime_h2o_automl, n_features = 10)
# plot_features(mod_lime)
# plot_explanations(mod_lime)
model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_h2o_automl <- predict_surrogate(
	explainer = explainer_h2o_automl, 
	new_observation = new_value, 
	n_features = 10, 
	n_permutations = 1000,
	type = "lime"
)
plot(lime_h2o_automl)

# ** Profiling ------------------------------------------------------------
cetp_h2o_automl <- predict_profile(
	explainer = explainer_h2o_automl, 
	new_observation = new_value,
	type = "ceteris_paribus",
	variables = c("hour", "wday")
)
cetp_h2o_automl
plot(cetp_h2o_automl, variables = c("hour", "wday"))

# ** Diagnostics ----------------------------------------------------------
preddiag_h2o_automl <- predict_diagnostics(
	explainer = explainer_h2o_automl, 
	new_observation = new_value,
	variables = c("hour", "wday")
)
preddiag_h2o_automl
plot(preddiag_h2o_automl, variables = c("hour", "wday"))



h2o.shutdown()
