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
		.smooth = FALSE, .title = ""
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
	mutate(index.num = standardize_vec(index.num)) |> 
	# fourier series
	tk_augment_fourier(date, .periods = rolling_periods, .K = 2) |> 
	rename_with(~ str_remove_all(., "date_")) |> 
	# drop na
	drop_na()
	

# * Train / Test Sets -----------------------------------------------------

splits <- m4_ts_prep |> time_series_split(assess = horizon, cumulative = TRUE)
splits |>
	tk_time_series_cv_plan() |>
	plot_time_series_cv_plan(date, value)



# Modelling ---------------------------------------------------------------

# * H2O Setup -------------------------------------------------------------

Sys.setenv(JAVA_HOME = "/usr/lib/jvm/jdk-17/")
h2o.init()

h2o.no_progress() 

# target <- "value"
# data_h2o <- as.h2o(training(splits) |> select(-date))
# 
# model_h2o_automl <- h2o.automl(
# 	y = target, 
# 	training_frame = data_h2o, 
# 	max_runtime_secs = 30, 
# 	max_models = 20
# )
# 
# leader_board <- h2o.get_leaderboard(model_h2o_automl)
# head(leader_board)


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
		max_runtime_secs = 30,
		max_runtime_secs_per_model = 30,
		max_models = 10,
		nfolds = 10,
		sort_metric = "rmse",
		verbosity = NULL,
		seed = 123
	)
model_spec_h2o

rcp_spec <- recipe(value ~ ., data = training(splits))


# * Workflows -------------------------------------------------------------

# - This step will take some time depending on your engine specifications
wrkfl_fit_h2o <- workflow() |>
	add_model(model_spec_h2o) |>
	add_recipe(rcp_spec) |>
	fit(training(splits))
wrkfl_fit_h2o

wrkfl_fit_h2o |> automl_leaderboard() |> head(20)


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

explainer_h2o_automl <- explain_h2o(
	model = wrkfl_fit_h2o$fit$fit, 
	data = testing(splits),
	y = testing(splits)$value,
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
	variable = "lag48", 
	type = "partial"
)
plot(pdp_h2o_automl)

# Accumulated Local Effect
ale_h2o_automl <- model_profile(
	explainer_h2o_automl,
	variable = "lag48", 
	type = "accumulated"
)
plot(ale_h2o_automl)


# * Prediction Understanding ----------------------------------------------
new_value <- testing(splits) |> filter(date == ymd_hms("2017-07-23 02:00:00"))
new_value <- testing(splits) |> filter(date == ymd_hms("2017-07-24 04:00:00"))

# Break Down
# https://medium.com/responsibleml/basic-xai-with-dalex-part-4-break-down-method-2cd4de43abdd
pb_h2o_automl <- predict_parts(
	explainer_h2o_automl,
	new_observation = new_value,
	type = "break_down"
)
plot(pb_h2o_automl)

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
# explainer_lime_h2o_automl <- lime(data_h2o, model_h2o_automl)
# mod_lime <- explain(new_date_birth, explainer_lime_h2o_automl, n_features = 8)
# plot_features(mod_lime)
# plot_explanations(mod_lime)

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_h2o_automl <- predict_surrogate(
	explainer = explainer_h2o_automl, 
	new_observation = new_value, 
	n_features = 8, 
	n_permutations = 100,
	type = "lime"
)
plot(lime_h2o_automl)



h2o.shutdown()
